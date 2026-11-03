"""Extra analysis and figures from saved AVeriTeC results (no API calls).

    python analyse_averitec.py results/dev/averitec_search_blocked_v2_dev.jsonl --out results/dev/analysis
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List
from urllib.parse import urlparse

import eval_averitec as ev

INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BAR, BAR_MUTED = "#256abf", "#9ec5f4"


def _list(v) -> List[str]:
    if isinstance(v, list):
        return v
    try:
        out = ast.literal_eval(v) if isinstance(v, str) else []
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def _acc(rows) -> str:
    if not rows:
        return "n/a"
    k = sum(bool(r["correct"]) for r in rows)
    return f"{100 * k / len(rows):.0f}% ({k}/{len(rows)})"


def _source_type(host: str) -> str:
    if "wikipedia" in host:
        return "Wikipedia"
    if re.search(r"facebook|twitter|x\.com|youtube|instagram|tiktok|reddit|t\.co$", host):
        return "Social media and video"
    if re.search(r"\.edu|\.ac\.|academic\.|pubmed|jamanetwork|nature\.com|sciencedirect|springer|thelancet|bmj\.com|medrxiv|arxiv", host):
        return "Academic"
    if re.search(r"\.gov|\.gob\.|gov\.|\.mil|who\.int|un\.org|europa\.eu|parliament|\.go\.|imf\.org|worldbank\.org", host):
        return "Government and intergovernmental"
    return "News and other"


def load(paths: List[str], data_path: str) -> List[dict]:
    data = ev.load(data_path)
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error"):
                    continue
                r = ev.add_derived_flags(r)
                r["_item"] = data[r["id"]]
                rows.append(r)
    return rows


def breakdown(rows, key_fn) -> Dict[str, list]:
    groups = defaultdict(list)
    for r in rows:
        keys = key_fn(r)
        for k in keys if isinstance(keys, list) else [keys]:
            groups[k].append(r)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def _length_band(r) -> str:
    n = len(r["claim"].split())
    return "under 15 words" if n < 15 else "15 to 24 words" if n < 25 else "25 words or more"


def analyse(rows: List[dict], out: str) -> None:
    os.makedirs(out, exist_ok=True)
    L = ["# Further analysis of AVeriTeC results", "",
         f"{len(rows)} scored claims from {', '.join(sorted({r['condition'] for r in rows}))}, "
         f"prompt version(s) {', '.join(sorted({r.get('prompt_version', 'v1') for r in rows}))}, "
         f"model {', '.join(sorted({r.get('model', '') for r in rows}))}. Error records are excluded.", ""]
    csv_rows = []

    # 1. committed accuracy
    sr = [r for r in rows if r["gold_label"] in ("Supported", "Refuted")]
    committed = [r for r in sr if r["pred"] in ("likely_true", "likely_false")]
    L += ["## 1. Telling true from false", "",
          "| Claims AVeriTeC labels Supported or Refuted | Value |", "|---|---|",
          f"| Claims | {len(sr)} |",
          f"| Pipeline committed to true or false | {len(committed)} ({100 * len(committed) / max(1, len(sr)):.0f}%) |",
          f"| Accuracy when it committed | {_acc(committed)} |",
          f"| Labelled misleading or unverifiable instead | {len(sr) - len(committed)} |",
          f"| Accuracy on all of these claims | {_acc(sr)} |", ""]

    # 2. breakdowns
    dims = [
        ("AVeriTeC claim type", lambda r: _list(r["_item"].get("claim_types")) or ["(none)"]),
        ("AVeriTeC fact-checking strategy", lambda r: _list(r["_item"].get("fact_checking_strategies")) or ["(none)"]),
        ("Country (ISO code)", lambda r: r["_item"].get("location_ISO_code") or "(none)"),
        ("Where the claim appeared", lambda r: r["_item"].get("reporting_source") or "(none)"),
        ("Claim length", _length_band),
        ("Gold label", lambda r: r["gold_label"]),
    ]
    L += ["## 2. Accuracy by claim characteristics", "",
          "Groups can overlap (a claim can have several types or strategies). "
          "Groups of fewer than 5 claims are listed but too small to interpret.", ""]
    for title, fn in dims:
        groups = breakdown(rows, fn)
        L += [f"**{title}**", "", "| Group | Claims | Accuracy |", "|---|---|---|"]
        for g, xs in groups.items():
            L.append(f"| {g} | {len(xs)} | {_acc(xs)} |")
            csv_rows.append({"breakdown": title, "group": g, "claims": len(xs),
                             "correct": sum(bool(r['correct']) for r in xs),
                             "accuracy": round(sum(bool(r['correct']) for r in xs) / len(xs), 4)})
        L.append("")

    # 3. calibration
    L += ["## 3. Stated confidence against accuracy", "", "| Confidence | Claims | Accuracy |", "|---|---|---|"]
    for c in ("high", "medium", "low"):
        xs = [r for r in rows if r.get("confidence") == c]
        L.append(f"| {c} | {len(xs)} | {_acc(xs)} |")
    L.append("")

    # 4. search effort and cost
    by_s = breakdown(rows, lambda r: str(int(r["trace"].get("web_search_requests", 0))))
    by_s = dict(sorted(by_s.items(), key=lambda kv: int(kv[0])))
    L += ["## 4. Search effort, cost and accuracy", "",
          "| Searches used | Claims | Mean input tokens | Mean cost (USD) | Accuracy |", "|---|---|---|---|---|"]
    for k, xs in by_s.items():
        tok = sum(r["trace"]["input_tokens"] for r in xs) / len(xs)
        cost = sum(ev.record_cost(r) for r in xs) / len(xs)
        L.append(f"| {k} | {len(xs)} | {tok:,.0f} | {cost:.3f} | {_acc(xs)} |")
    total_tokens = sum(r["trace"]["input_tokens"] for r in rows)
    heavy = [r for r in rows if r["trace"].get("web_search_requests", 0) >= 4]
    if total_tokens and heavy:
        share = sum(r["trace"]["input_tokens"] for r in heavy) / total_tokens
        L += ["", f"Claims using 4 or more searches were {100 * len(heavy) / len(rows):.0f}% of claims but "
              f"{100 * share:.0f}% of input tokens. Input tokens rise faster than the number of searches "
              "because each further search re-reads everything gathered so far."]
    L += ["", f"Estimated total cost of these {len(rows)} claims: US${sum(ev.record_cost(r) for r in rows):.2f}.", ""]

    # 5. sources
    types, urls = Counter(), 0
    for r in rows:
        for u in r["trace"].get("retrieved_urls", []):
            types[_source_type(urlparse(u).netloc.lower().removeprefix("www."))] += 1
            urls += 1
    cited = [r for r in rows if r.get("sources")]
    grounded = sum(sum(u in r["trace"]["retrieved_urls"] for u in r["sources"]) for r in cited)
    n_cited = sum(len(r["sources"]) for r in cited)
    L += ["## 5. Sources", "", "| Type of site retrieved | Share of retrieved pages |", "|---|---|"]
    for t, v in types.most_common():
        L.append(f"| {t} | {100 * v / max(1, urls):.0f}% |")
    L += ["", f"{urls:,} pages retrieved in total, {urls / max(1, len(rows)):.1f} per claim. "
          f"Claims cited {n_cited / max(1, len(rows)):.1f} sources on average. "
          f"{grounded} of {n_cited} cited URLs ({100 * grounded / max(1, n_cited):.1f}%) were among the pages "
          "the search actually returned; the rest may be invented or recalled from training.", ""]

    # 6. distributions
    pc, gc = Counter(r["pred"] for r in rows), Counter(r["gold_mapped"] for r in rows)
    L += ["## 6. Predicted against gold label distribution", "", "| Label | Predicted | Gold |", "|---|---|---|"]
    for lab in ev.PRED_LABELS:
        L.append(f"| {lab} | {pc[lab]} | {gc[lab]} |")
    L.append("")

    with open(os.path.join(out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "analysis.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["breakdown", "group", "claims", "correct", "accuracy"])
        w.writeheader()
        w.writerows(csv_rows)

    figs = _figures(rows, breakdown(rows, dims[0][1]), by_s, out)
    print(f"Analysis written to {out}" + ("" if figs else " (install matplotlib for the figures)"))


def _style(ax, fig):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)


def _figures(rows, by_type, by_search, out) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    # accuracy by claim type (horizontal bars, n in the label)
    items = [(k, xs) for k, xs in by_type.items() if len(xs) >= 5]
    if items:
        items.sort(key=lambda kv: sum(bool(r["correct"]) for r in kv[1]) / len(kv[1]))
        fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(items) + 1.2), dpi=200)
        vals = [100 * sum(bool(r["correct"]) for r in xs) / len(xs) for _, xs in items]
        ax.barh(range(len(items)), vals, color=BAR, height=0.55)
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(items)), [f"{k} (n={len(xs)})" for k, xs in items])
        ax.set_xlim(0, 105)
        ax.set_xlabel("Accuracy (%)", fontsize=9, color=INK_2)
        ax.set_title("Accuracy by AVeriTeC claim type", fontsize=10, color=INK, loc="left")
        _style(ax, fig)
        ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "accuracy_by_claim_type.png"), facecolor=SURFACE)
        plt.close(fig)

    # input tokens and accuracy by number of searches
    ks = list(by_search)
    if ks:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=200)
        toks = [sum(r["trace"]["input_tokens"] for r in by_search[k]) / len(by_search[k]) / 1000 for k in ks]
        accs = [100 * sum(bool(r["correct"]) for r in by_search[k]) / len(by_search[k]) for k in ks]
        a1.bar(ks, toks, color=BAR, width=0.6)
        a2.bar(ks, accs, color=BAR_MUTED, width=0.6)
        for ax, vals, fmt in ((a1, toks, "{:.0f}k"), (a2, accs, "{:.0f}%")):
            for x, v in zip(ks, vals):
                ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
            ax.set_xlabel("Web searches used", fontsize=9, color=INK_2)
            _style(ax, fig)
        a1.set_title("Mean input tokens per claim", fontsize=10, color=INK, loc="left")
        a2.set_title("Accuracy (%)", fontsize=10, color=INK, loc="left")
        a2.set_ylim(0, 105)
        a2.set_xticks(range(len(ks)), [f"{k}\n(n={len(by_search[k])})" for k in ks])
        a1.set_xticks(range(len(ks)), ks)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "searches_tokens_accuracy.png"), facecolor=SURFACE)
        plt.close(fig)

    # committed vs hedged on Supported/Refuted claims
    sr = [r for r in rows if r["gold_label"] in ("Supported", "Refuted")]
    if sr:
        committed = [r for r in sr if r["pred"] in ("likely_true", "likely_false")]
        right = sum(bool(r["correct"]) for r in committed)
        parts = [("Committed, correct", right, BAR), ("Committed, wrong", len(committed) - right, "#c33d3d"),
                 ("Misleading or unverifiable", len(sr) - len(committed), "#b8b7b1")]
        fig, ax = plt.subplots(figsize=(6.4, 1.9), dpi=200)
        left = 0
        for label, v, col in parts:
            ax.barh([0], [v], left=left, color=col, height=0.5, edgecolor=SURFACE, linewidth=2)
            if v:
                ax.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=9,
                        color="#ffffff" if col != "#b8b7b1" else INK)
            left += v
        ax.set_yticks([])
        ax.set_xlim(0, len(sr))
        ax.set_title(f"Claims labelled Supported or Refuted (n={len(sr)})", fontsize=10, color=INK, loc="left")
        ax.legend([p[0] for p in parts], loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3,
                  frameon=False, fontsize=8)
        _style(ax, fig)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "committed_verdicts.png"), facecolor=SURFACE)
        plt.close(fig)
    return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Further analysis of AVeriTeC results (no API calls).")
    ap.add_argument("files", nargs="+", help="Results files to analyse together.")
    ap.add_argument("--data", default=ev.DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(ev.RESULTS_DIR, "analysis"))
    args = ap.parse_args(argv)
    analyse(load(args.files, args.data), args.out)


if __name__ == "__main__":
    main()
