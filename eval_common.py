"""Shared helpers for the stage 2 and stage 3 evaluations."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import eval_averitec as ev

RESULTS_DIR = ev.RESULTS_DIR
_lock = threading.Lock()

CATEGORY_NAMES = {
    "content_manipulation": "Content manipulation",
    "miscontextualised": "Miscontextualised",
    "authentic_truthful": "Authentic and truthful",
    "unverifiable": "Unverifiable",
}


# inputs


def read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        sys.exit(f"{path} not found.")
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_ids(paths: Iterable[str]) -> set:
    return ev._read_ids(paths)


def done_ids(path: str, retry_errors: bool) -> set:
    return ev._done_ids(path, retry_errors)


def load_results(paths: Iterable[str]) -> Dict[str, List[dict]]:
    """Latest successful record per (condition, id)."""
    by_cond: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prev = by_cond[rec["condition"]].get(rec["id"])
                if prev is None or prev.get("error") or not rec.get("error"):
                    by_cond[rec["condition"]][rec["id"]] = rec
    return {c: sorted(d.values(), key=lambda r: str(r["id"])) for c, d in by_cond.items()}


def results_files(prefix: str) -> List[str]:
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted(os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR)
                  if f.startswith(prefix) and f.endswith(".jsonl"))


# cost


def token_cost(trace: dict, model: str) -> float:
    p_in, p_out = ev.model_prices(model)
    return (trace.get("input_tokens", 0) * p_in + trace.get("output_tokens", 0) * p_out) / 1e6


def confirm_cost(n: int, per_item: float, basis: str, args, free_note: str = "") -> bool:
    total = n * per_item
    print(f"Estimated cost: US${total:.2f} for {n} items (about US${per_item:.3f} each, {basis}).{free_note}")
    if getattr(args, "budget", None) is not None:
        print(f"The run will stop once about US${args.budget:.2f} has been spent.")
    if getattr(args, "dry_run", False):
        return False
    if per_item == 0:
        return True
    if not ev._confirm("Go ahead? [y/N] ", getattr(args, "yes", False)):
        print("Cancelled; nothing was sent.")
        return False
    return True


def observed_usage(prefix: str, condition: str) -> Optional[dict]:
    rows = []
    for path in results_files(prefix):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("condition") == condition and not r.get("error") and r.get("trace"):
                    rows.append(r["trace"])
    if len(rows) < 5:
        return None
    keys = ("input_tokens", "output_tokens", "serpapi_searches")
    return {k: sum(t.get(k, 0) for t in rows) / len(rows) for k in keys} | {"n": len(rows)}


# running


class Writer:
    def __init__(self, path: str, total: int, budget: Optional[float]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.total, self.budget = total, budget
        self.completed = self.errors = 0
        self.spent = 0.0

    def write(self, rec: dict) -> None:
        with _lock:
            self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self.f.flush()
            self.completed += 1
            self.errors += bool(rec.get("error"))
            self.spent += rec.get("cost_usd", 0.0) or 0.0
        mark = "ERR" if rec.get("error") else ("ok " if rec.get("correct") else "x  ")
        print(f"[{self.completed}/{self.total}] {mark} {str(rec['id']):<12} gold={rec.get('gold', ''):<21} "
              f"pred={str(rec.get('pred')):<21} US${self.spent:6.2f}"
              + (f"  {rec['error'][:70]}" if rec.get("error") else ""))

    def over_budget(self) -> bool:
        return self.budget is not None and self.spent >= self.budget

    def close(self):
        self.f.close()


def run_items(items: List[dict], fn, workers: int, writer: Writer) -> None:
    """Run fn(item) -> record in parallel; stop on billing errors or budget."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fn, it) for it in items]
        try:
            for fut in as_completed(futures):
                rec = fut.result()
                if rec.get("error") and ev.FATAL_ERROR_RE.search(rec["error"]):
                    pool.shutdown(wait=True, cancel_futures=True)
                    writer.close()
                    sys.exit(f"\nStopped: {rec['error'][:200]}\n{writer.completed} items were saved. "
                             "Fix the problem, then run the same command again to carry on.")
                writer.write(rec)
                if writer.over_budget():
                    pool.shutdown(wait=True, cancel_futures=True)
                    writer.close()
                    sys.exit(f"\nStopped: budget reached (about US${writer.spent:.2f}). "
                             f"{writer.completed} items saved; run again to carry on.")
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            writer.close()
            print("\nStopping; finished items are saved. Run the same command again to resume.")
            raise SystemExit(130)


# metrics


def prf(records: List[dict], labels: List[str]) -> Dict[str, dict]:
    out = {}
    for lab in labels:
        tp = sum(r["pred"] == lab and r["gold"] == lab for r in records)
        fp = sum(r["pred"] == lab and r["gold"] != lab for r in records)
        fn = sum(r["pred"] != lab and r["gold"] == lab for r in records)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0, "support": tp + fn}
    return out


def accuracy(records) -> float:
    return sum(bool(r["correct"]) for r in records) / len(records) if records else float("nan")


def macro_f1(records, labels) -> float:
    """Macro F1 over every category that is either correct or predicted."""
    labels = list(dict.fromkeys(list(labels) + [r["pred"] for r in records if r.get("pred")]))
    per = prf(records, labels)
    present = [l for l in labels if per[l]["support"] > 0 or any(r["pred"] == l for r in records)]
    return sum(per[l]["f1"] for l in present) / len(present) if present else float("nan")


def summarise(records: List[dict], gold_labels: List[str], pred_labels: List[str]) -> dict:
    ok = [r for r in records if not r.get("error")]
    conf = {g: {p: 0 for p in pred_labels} for g in gold_labels}
    for r in ok:
        if r["gold"] in conf and r["pred"] in conf[r["gold"]]:
            conf[r["gold"]][r["pred"]] += 1
    by_conf = {c: [r for r in ok if r.get("confidence") == c] for c in ("high", "medium", "low")}
    return {
        "n_total": len(records),
        "n_scored": len(ok),
        "n_errors": len(records) - len(ok),
        "accuracy": accuracy(ok),
        "accuracy_ci95": ev._bootstrap_ci(ok, accuracy),
        "macro_f1": macro_f1(ok, gold_labels),
        "macro_f1_ci95": ev._bootstrap_ci(ok, lambda rs: macro_f1(rs, gold_labels)),
        "per_class": prf(ok, pred_labels),
        "confusion": conf,
        "coverage": sum(r["pred"] != "unverifiable" for r in ok) / len(ok) if ok else float("nan"),
        "by_confidence": {c: {"n": len(v), "accuracy": accuracy(v) if v else None} for c, v in by_conf.items()},
        "mean_cost_usd": sum(r.get("cost_usd", 0) or 0 for r in ok) / len(ok) if ok else 0.0,
        "total_cost_usd": sum(r.get("cost_usd", 0) or 0 for r in records),
        "mean_input_tokens": sum((r.get("trace") or {}).get("input_tokens", 0) for r in ok) / len(ok) if ok else 0.0,
        "serpapi_searches": sum((r.get("trace") or {}).get("serpapi_searches", 0) for r in ok),
    }


def paired(by_cond: Dict[str, List[dict]]) -> List[dict]:
    rows = []
    conds = list(by_cond)
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            ra = {r["id"]: r for r in by_cond[a] if not r.get("error")}
            rb = {r["id"]: r for r in by_cond[b] if not r.get("error")}
            common = sorted(set(ra) & set(rb), key=str)
            if not common:
                continue
            only_a = sum(ra[k]["correct"] and not rb[k]["correct"] for k in common)
            only_b = sum(rb[k]["correct"] and not ra[k]["correct"] for k in common)
            rows.append({"a": a, "b": b, "n_common": len(common),
                         "acc_a": accuracy([ra[k] for k in common]), "acc_b": accuracy([rb[k] for k in common]),
                         "only_a_correct": only_a, "only_b_correct": only_b,
                         "mcnemar_p": ev._mcnemar_exact(only_a, only_b)})
    return rows


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def ci(c: Tuple[float, float]) -> str:
    return "n/a" if math.isnan(c[0]) else f"{100 * c[0]:.1f} to {100 * c[1]:.1f}"


# figures

_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def plot_confusion(conf: dict, row_labels: List[str], col_labels: List[str], title: str,
                   xlabel: str, ylabel: str, path: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return False
    rows, cols = list(conf), list(next(iter(conf.values())))
    counts = [[conf[r][c] for c in cols] for r in rows]
    shares = [[v / sum(row) if sum(row) else 0.0 for v in row] for row in counts]
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 2.4, 0.85 * len(rows) + 1.6), dpi=200)
    ax.imshow(shares, cmap=LinearSegmentedColormap.from_list("seq", _RAMP), vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(counts):
        for j, v in enumerate(row):
            s = shares[i][j]
            ax.text(j, i, f"{v}\n{100 * s:.0f}%", ha="center", va="center", fontsize=9,
                    color="#ffffff" if s > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(cols)), col_labels, fontsize=8.5)
    ax.set_yticks(range(len(rows)), row_labels, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9, color="#52514e")
    ax.set_ylabel(ylabel, fontsize=9, color="#52514e")
    ax.set_title(title + "\n(cell shade = share of row)", fontsize=10, color="#0b0b0b")
    ax.tick_params(length=0, colors="#52514e")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def sample(items: List[dict], n: Optional[int], seed: int) -> List[dict]:
    """Stratified on 'label', same method as the AVeriTeC runs."""
    for i, it in enumerate(items):
        it.setdefault("_id", i)
    return ev.stratified_sample(items, n, seed)


def rng(seed: int) -> random.Random:
    return random.Random(seed)
