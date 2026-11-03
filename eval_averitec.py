"""Stage 1 evaluation on AVeriTeC (also used for the post-cutoff claims).

Conditions: search_blocked, search_open, no_search.

    python eval_averitec.py download
    python eval_averitec.py run --condition search_blocked --sample 150 --batch --dry-run
    python eval_averitec.py run --condition search_blocked --sample 150 --batch
    python eval_averitec.py run --condition no_search --sample 150 --batch
    python eval_averitec.py score
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import factcheck as fc

DEV_URL = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json"
DEFAULT_DATA = os.path.join("data", "averitec_dev.json")
RESULTS_DIR = "results"

GOLD_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]
PRED_LABELS = ["likely_true", "likely_false", "misleading", "unverifiable"]
LABEL_MAP = {
    "Supported": "likely_true",
    "Refuted": "likely_false",
    "Not Enough Evidence": "unverifiable",
    # fourth class maps to "misleading"
    "Conflicting Evidence/Cherrypicking": "misleading",
}
CONDITIONS = ("search_blocked", "search_open", "no_search")

# Hidden from search in the search_blocked condition (max 64 entries).
BLOCKED_DOMAINS = [
    # dedicated fact-checkers
    "politifact.com", "factcheck.org", "snopes.com", "leadstories.com",
    "checkyourfact.com", "africacheck.org", "factcheck.afp.com", "fullfact.org",
    "misbar.com", "newsmeter.in", "altnews.in", "newsmobile.in", "dubawa.org",
    "polygraph.info", "healthfeedback.org", "sciencefeedback.co", "boomlive.in",
    "vishvasnews.com", "icirnigeria.org", "verafiles.org", "ghanafact.com",
    "zimfact.org", "factcheckni.org", "covid19facts.ca", "thelogicalindian.com",
    "truthorfiction.com", "factually.co", "factcheck.thecable.ng",
    "toolbox.google.com/factcheck", "aap.com.au", "factly.in", "maldita.es", "newtral.es", "correctiv.org",
    "logically.ai", "teyit.org", "facta.news",
    # fact-check sections of news outlets
    "usatoday.com/story/news/factcheck", "indiatoday.in/fact-check",
    "thequint.com/news/webqoof", "rappler.com/newsbreak/fact-check",
    "thegazette.com/factchecker", "factcheck.thedispatch.com", "poynter.org",
    "reuters.com/fact-check", "apnews.com/hub/ap-fact-check",
    "apnews.com/article/fact-check", "abc.net.au/news/factcheck",
    "oneindia.com/fact-check", "republicworld.com/fact-check",
    "washingtonpost.com/politics/2020/*/fact-checker",
    "bbc.co.uk/news/reality_check",
    # archives and dataset mirrors
    "archive.org", "archive.ph", "archive.today", "github.com",
    "raw.githubusercontent.com", "huggingface.co", "kaggle.com",
    "paperswithcode.com", "fever.ai", "arxiv.org", "aclanthology.org",
]
assert len(BLOCKED_DOMAINS) < 64, "the web search blocklist should stay under 64 entries"

# Fact-check pages on unblocked sites, flagged at score time.
FACTCHECK_URL_RE = re.compile(
    r"fact[-_]?check|factcheck|/verify/|webqoof|reality[-_]check|debunk|hoax", re.I
)
# Fact-checker names to look for in justifications.
FACTCHECKER_NAMES_RE = re.compile(
    r"politifact|snopes|factcheck\.org|full ?fact|africa ?check|vera ?files|lead ?stories|"
    r"afp fact|aap fact|check ?your ?fact|boom ?live|alt ?news|dubawa|misbar|health ?feedback|"
    r"logically|fact[- ]?check(?:er|ers|ing)? (?:organi[sz]ation|site|outlet)|"
    r"(?:rated|ruled|labell?ed) (?:it |the claim )?(?:false|misleading|mostly)",
    re.I,
)

_write_lock = threading.Lock()

# Errors that would hit every remaining claim, so the run stops.
FATAL_ERROR_RE = re.compile(
    r"credit balance|AuthenticationError|PermissionDeniedError|invalid x-api-key|"
    r"ANTHROPIC_API_KEY is not set|SerpApi unavailable",
    re.I,
)


# Data


def download(dest: str) -> None:
    import requests

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"Downloading AVeriTeC dev set to {dest} ...")
    r = requests.get(DEV_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Saved {len(data)} claims: {dict(Counter(x['label'] for x in data))}")


def load(path: str) -> List[dict]:
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run `python eval_averitec.py download` first.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        item["_id"] = i  # position in the file is the stable claim id
    return data


def stratified_sample(data: List[dict], n: Optional[int], seed: int) -> List[dict]:
    """Pick n claims keeping the label proportions (same n and seed, same claims)."""
    if not n or n >= len(data):
        return list(data)
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)
    quotas = {lab: n * len(items) / len(data) for lab, items in by_label.items()}
    alloc = {lab: int(q) for lab, q in quotas.items()}
    for lab in sorted(quotas, key=lambda l: quotas[l] - alloc[l], reverse=True)[: n - sum(alloc.values())]:
        alloc[lab] += 1
    rng = random.Random(seed)
    chosen = []
    for lab in sorted(by_label):
        chosen.extend(rng.sample(by_label[lab], alloc[lab]))
    return sorted(chosen, key=lambda x: x["_id"])


def _format_date(raw: Optional[str]) -> Optional[str]:
    try:
        return dt.datetime.strptime(raw, "%d-%m-%Y").strftime("%-d %B %Y" if os.name != "nt" else "%#d %B %Y")
    except (TypeError, ValueError):
        return raw or None


def build_context(item: dict) -> Optional[str]:
    """Claim metadata (speaker, date, platform, country) passed to the model."""
    parts = []
    if item.get("speaker"):
        parts.append(f"Speaker: {item['speaker']}")
    date = _format_date(item.get("claim_date"))
    if date:
        parts.append(f"Date the claim was made: {date}")
    if item.get("reporting_source"):
        parts.append(f"Where it appeared: {item['reporting_source']}")
    if item.get("location_ISO_code"):
        parts.append(f"Country code: {item['location_ISO_code']}")
    return "\n".join(parts) or None


# Leakage detection


def _norm_url(url: str) -> str:
    """Reduce a URL to host and path so variants of the same article match."""
    if not url:
        return ""
    m = re.search(r"web\.archive\.org/web/[^/]+/(.*)", url)
    if m:
        url = m.group(1)
    if "://" not in url:
        url = "http://" + url
    p = urlparse(url)
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{p.path.rstrip('/')}".lower()


def _matches_block(url: str, entry: str) -> bool:
    norm = _norm_url(url)
    host, _, path = norm.partition("/")
    e_host, _, e_path = entry.lower().partition("/")
    if not (host == e_host or host.endswith("." + e_host)):
        return False
    if not e_path:
        return True
    pattern = "^" + re.escape(e_path).replace(r"\*", "[^/]*")
    return re.match(pattern, path) is not None


def leakage_flags(item: dict, retrieved: Iterable[str], cited: Iterable[str]) -> dict:
    target = _norm_url(item.get("fact_checking_article") or "")
    retrieved, cited = list(retrieved), list(cited)
    return {
        # own fact-check article
        "leak_exact": bool(target) and any(_norm_url(u) == target for u in retrieved + cited),
        # any fact-checking site
        "leak_factcheck_site": any(_matches_block(u, e) for u in retrieved + cited for e in BLOCKED_DOMAINS),
    }


# Cost

# USD per million tokens (input, output), checked 28 Sep 2026. Batch halves both;
# web search is USD 10 per 1,000 searches.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
SEARCH_PRICE = 10.0 / 1000
BATCH_DISCOUNT = 0.5

# Per-claim usage from the development run, for estimates before any results exist.
DEFAULT_USAGE = {
    "search": {"input_tokens": 48500, "output_tokens": 1700, "web_search_requests": 2.6},
    "no_search": {"input_tokens": 2500, "output_tokens": 900, "web_search_requests": 0.0},
}
# share of input tokens that is not search results
_BASE_INPUT_TOKENS = 3000


def model_prices(model: str) -> Tuple[float, float]:
    for prefix, price in PRICES.items():
        if model.startswith(prefix):
            return price
    return PRICES["claude-sonnet-5"]  # unknown model: assume Sonnet pricing and say so in the estimate


def record_cost(r: dict) -> float:
    """Estimated USD cost of one record from its token and search counts."""
    t = r.get("trace") or {}
    p_in, p_out = model_prices(r.get("model", fc.MODEL))
    b_in, b_out = r.get("batch_tokens") or (0, 0)
    full_in = max(0, t.get("input_tokens", 0) - b_in)
    full_out = max(0, t.get("output_tokens", 0) - b_out)
    return (
        (full_in * p_in + full_out * p_out) / 1e6
        + (b_in * p_in + b_out * p_out) * BATCH_DISCOUNT / 1e6
        + t.get("web_search_requests", 0) * SEARCH_PRICE
    )


def _observed_traces(base_condition: str) -> List[dict]:
    rows = []
    if not os.path.isdir(RESULTS_DIR):
        return rows
    for root, _dirs, files in os.walk(RESULTS_DIR):
        for name in files:
            if not (name.startswith(f"averitec_{base_condition}") and name.endswith(".jsonl")):
                continue
            with open(os.path.join(root, name), encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not r.get("error") and r.get("trace"):
                        rows.append(r["trace"])
    return rows


def _usage_with_cap(traces: List[dict], cap: Optional[int]) -> dict:
    """Mean usage per claim if searches had been capped at `cap`."""
    keys = ("input_tokens", "output_tokens", "web_search_requests")
    by_count: Dict[int, List[dict]] = defaultdict(list)
    for t in traces:
        by_count[int(t.get("web_search_requests", 0))].append(t)
    capped = []
    for t in traces:
        n = int(t.get("web_search_requests", 0))
        if cap and n > cap and by_count.get(cap):
            ref = by_count[cap]
            capped.append({k: sum(x.get(k, 0) for x in ref) / len(ref) for k in keys})
        else:
            capped.append(t)
    return {k: sum(t.get(k, 0) for t in capped) / len(capped) for k in keys}


def estimate_cost(n: int, base_condition: str, max_searches: Optional[int], batch: bool) -> Tuple[float, str]:
    key = "no_search" if base_condition == "no_search" else "search"
    traces = _observed_traces(base_condition)
    if len(traces) < 10:
        traces = _observed_traces("search_blocked" if key == "search" else "no_search")
    if len(traces) >= 10:
        usage = _usage_with_cap(traces, max_searches if key == "search" else None)
        basis = f"from {len(traces)} earlier claims"
        if key == "search" and max_searches:
            basis += f", with at most {max_searches} searches"
    else:
        usage = dict(DEFAULT_USAGE[key])
        basis = "from the 28 September run"
        if key == "search" and max_searches and usage["web_search_requests"] > max_searches:
            ratio = max_searches / usage["web_search_requests"]
            usage["input_tokens"] = _BASE_INPUT_TOKENS + (usage["input_tokens"] - _BASE_INPUT_TOKENS) * ratio
            usage["web_search_requests"] = max_searches
            basis += f", scaled to at most {max_searches} searches"
    p_in, p_out = model_prices(fc.MODEL)
    factor = BATCH_DISCOUNT if batch else 1.0
    per_claim = (usage["input_tokens"] * p_in + usage["output_tokens"] * p_out) * factor / 1e6 \
        + usage["web_search_requests"] * SEARCH_PRICE
    if not any(fc.MODEL.startswith(k) for k in PRICES):
        basis += f"; no price listed for {fc.MODEL}, Sonnet 5 prices assumed"
    return per_claim * n, f"about US${per_claim:.3f} per claim ({basis})"


# Run


def results_path(label: str) -> str:
    return os.path.join(RESULTS_DIR, f"averitec_{label}.jsonl")


def _read_ids(paths: Iterable[str]) -> set:
    ids = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def _done_ids(path: str, retry_errors: bool) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by a crash
            if retry_errors and rec.get("error"):
                continue
            done.add(rec["id"])
    return done


def evaluate_one(item: dict, args, first_response=None) -> dict:
    """Check one claim and build its results record."""
    trace = fc.new_trace()
    started = time.time()
    use_context = not args.no_context
    record = {
        "id": item["_id"],
        "condition": args.label,
        "base_condition": args.condition,
        "model": fc.MODEL,
        "prompt_version": getattr(fc, "PROMPT_VERSION", "v1"),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "claim": item["claim"],
        "gold_label": item["label"],
        "gold_mapped": LABEL_MAP[item["label"]],
        "used_context": use_context,
        "max_searches": args.max_searches if args.condition != "no_search" else 0,
        "batch": first_response is not None,
        "error": None,
    }
    if first_response is not None:
        u = getattr(first_response, "usage", None)
        record["batch_tokens"] = [getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0]
    try:
        v = fc.verify_claim(
            item["claim"],
            context=build_context(item) if use_context else None,
            use_search=args.condition != "no_search",
            blocked_domains=BLOCKED_DOMAINS if args.condition == "search_blocked" else None,
            trace=trace,
            max_searches=args.max_searches,
            first_response=first_response,
        )
        record.update(
            pred=v.verdict,
            confidence=v.confidence,
            justification=v.justification,
            sources=v.sources,
            correct=v.verdict == record["gold_mapped"],
            **leakage_flags(item, trace["retrieved_urls"], v.sources),
        )
    except Exception as exc:
        record.update(pred=None, correct=None, error=f"{type(exc).__name__}: {exc}")
    record["elapsed_s"] = round(time.time() - started, 2)
    record["trace"] = trace
    record["cost_usd"] = round(record_cost(record), 5)
    return record


def _check_prompt_version(path: str) -> None:
    current = getattr(fc, "PROMPT_VERSION", "v1")
    if not os.path.exists(path):
        return
    old = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                old.add(json.loads(line).get("prompt_version", "v1"))
            except json.JSONDecodeError:
                continue
    if old - {current}:
        sys.exit(
            f"{path} holds results from prompt version(s) {sorted(old - {current})}, but the "
            f"current prompt is {current}. Move that file somewhere else (for example "
            f"results\\dev) so old and new results are not mixed, then run again."
        )


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Not running: add --yes to confirm the cost when running without a terminal.")
        return False
    return input(prompt).strip().lower() in ("y", "yes")


class _Writer:
    """Appends records to the results file and prints progress."""

    def __init__(self, out, total: int, budget: Optional[float]):
        self.out, self.total, self.budget = out, total, budget
        self.completed = self.errors = 0
        self.spent = 0.0

    def write(self, rec: dict) -> None:
        with _write_lock:
            self.out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.out.flush()
            self.completed += 1
            self.errors += bool(rec["error"])
            self.spent += rec.get("cost_usd", 0.0)
        mark = "ERR" if rec["error"] else ("ok " if rec["correct"] else "x  ")
        print(
            f"[{self.completed}/{self.total}] {mark} #{rec['id']:<3} gold={rec['gold_mapped']:<13} "
            f"pred={str(rec.get('pred')):<13} US${self.spent:6.2f}"
            + (f"  {rec['error'][:80]}" if rec["error"] else "")
        )

    def over_budget(self) -> bool:
        return self.budget is not None and self.spent >= self.budget


def _finish(pool_items, args, writer, first_responses=None) -> None:
    """Run claims in parallel, stopping on fatal errors or at the budget."""
    first_responses = first_responses or {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_one, it, args, first_responses.get(it["_id"])): it for it in pool_items}
        try:
            for fut in as_completed(futures):
                rec = fut.result()
                if rec["error"] and FATAL_ERROR_RE.search(rec["error"]):
                    pool.shutdown(wait=True, cancel_futures=True)
                    sys.exit(
                        f"\nStopped: {rec['error'][:200]}\n{writer.completed} claims were saved. Fix the "
                        f"problem (for example top up your API credit), then run the same command again."
                    )
                writer.write(rec)
                if writer.over_budget():
                    pool.shutdown(wait=True, cancel_futures=True)
                    sys.exit(
                        f"\nStopped: the --budget of US${args.budget:.2f} was reached "
                        f"(about US${writer.spent:.2f} spent). {writer.completed} claims were saved; "
                        f"run the same command again to carry on."
                    )
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print("\nStopping; finished claims are saved. Run the same command again to resume.")
            raise SystemExit(130)


def _batch_state_path(path: str) -> str:
    return path.replace(".jsonl", ".batch.json")


def _run_batch(todo: List[dict], args, path: str, writer: _Writer) -> None:
    """Send each claim's first request through the Batch API, then finish any that need more turns."""
    client = fc.get_client()
    state_path = _batch_state_path(path)
    by_id = {it["_id"]: it for it in todo}

    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        print(f"Resuming batch {state['batch_id']} ({len(state['ids'])} claims) instead of submitting a new one.")
    else:
        requests_ = []
        for it in todo:
            req = fc.build_claim_request(
                it["claim"],
                context=None if args.no_context else build_context(it),
                use_search=args.condition != "no_search",
                blocked_domains=BLOCKED_DOMAINS if args.condition == "search_blocked" else None,
                max_searches=args.max_searches,
            )
            requests_.append({"custom_id": f"claim-{it['_id']}", "params": req})
        try:
            batch = client.messages.batches.create(requests=requests_)
        except Exception as exc:
            sys.exit(f"Could not submit the batch: {exc}")
        state = {"batch_id": batch.id, "ids": [it["_id"] for it in todo],
                 "submitted": dt.datetime.now().isoformat(timespec="seconds")}
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        print(f"Submitted batch {batch.id} with {len(todo)} claims. Batches usually finish within an hour "
              f"and always within 24 hours.")

    batch_id = state["batch_id"]
    try:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            c = batch.request_counts
            print(f"  {dt.datetime.now():%H:%M:%S}  {batch.processing_status}: {c.succeeded} done, "
                  f"{c.processing} waiting, {c.errored} errored, {c.expired} expired", flush=True)
            if batch.processing_status == "ended":
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopped waiting. The batch keeps running on Anthropic's side; run the same command "
              "again later to collect the results (nothing is charged twice).")
        raise SystemExit(130)

    first_responses, failed = {}, []
    for entry in client.messages.batches.results(batch_id):
        try:
            cid = int(entry.custom_id.split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        if cid not in by_id:
            continue
        if entry.result.type == "succeeded":
            first_responses[cid] = entry.result.message
        else:
            detail = getattr(getattr(entry.result, "error", None), "error", None)
            failed.append((cid, f"batch request {entry.result.type}: {getattr(detail, 'message', '') or detail or ''}"))

    for cid, err in failed:
        rec = {"id": cid, "condition": args.label, "base_condition": args.condition, "model": fc.MODEL,
               "prompt_version": getattr(fc, "PROMPT_VERSION", "v1"), "claim": by_id[cid]["claim"],
               "gold_label": by_id[cid]["label"], "gold_mapped": LABEL_MAP[by_id[cid]["label"]],
               "pred": None, "correct": None, "error": err, "batch": True, "trace": fc.new_trace(),
               "elapsed_s": 0.0, "cost_usd": 0.0}
        if FATAL_ERROR_RE.search(err):
            sys.exit(f"Stopped: {err[:200]}\nFix the problem, then run the same command again.")
        writer.write(rec)

    ready = [by_id[cid] for cid in state["ids"] if cid in first_responses]
    print(f"Batch finished: {len(ready)} replies to process, {len(failed)} failed "
          f"(re-run later with --retry-errors).")
    _finish(ready, args, writer, first_responses)
    os.remove(state_path)


def run(args) -> None:
    args.label = args.condition + (f"_{args.tag}" if args.tag else "")
    data = load(args.data)
    if args.only:
        keep = _read_ids(args.only)
        data = [x for x in data if x["_id"] in keep]
        print(f"Restricted to the {len(data)} claims in {', '.join(args.only)}.")
    if args.exclude:
        excluded = _read_ids(args.exclude)
        data = [x for x in data if x["_id"] not in excluded]
        print(f"Excluding {len(excluded)} claims already used in {', '.join(args.exclude)}.")
    items = stratified_sample(data, args.sample, args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = results_path(args.label)
    _check_prompt_version(path)
    done = _done_ids(path, args.retry_errors)
    todo = [it for it in items if it["_id"] not in done]
    resuming_batch = os.path.exists(_batch_state_path(path))

    print(
        f"{args.label}: {len(items)} claims selected ({dict(Counter(i['label'] for i in items))}), "
        f"{len(items) - len(todo)} already done, {len(todo)} to run. Model {fc.MODEL}"
        + (f", at most {args.max_searches} searches per claim" if args.condition != "no_search" else "")
        + (", Batch API" if args.batch else "") + "."
    )
    if not todo:
        return

    total, basis = estimate_cost(len(todo), args.condition, args.max_searches, args.batch)
    print(f"Estimated cost: US${total:.2f}, {basis}.")
    if args.budget is not None and not args.batch:
        print(f"The run will stop once about US${args.budget:.2f} has been spent.")
    if args.dry_run:
        return
    if not resuming_batch and not _confirm("Go ahead? [y/N] ", args.yes):
        print("Cancelled; nothing was sent.")
        return

    fc.get_client()  # stop now if the API key is missing
    with open(path, "a", encoding="utf-8") as out:
        writer = _Writer(out, len(todo), None if args.batch else args.budget)
        if args.batch:
            _run_batch(todo, args, path, writer)
        else:
            _finish(todo, args, writer)
    print(f"Done. {writer.errors} errors, about US${writer.spent:.2f} spent. Results in {path}")
    if writer.errors:
        print("Re-run with --retry-errors to try the failed claims again.")


# Metrics


def _prf(records: List[dict]) -> Dict[str, dict]:
    out = {}
    for lab in PRED_LABELS:
        tp = sum(r["pred"] == lab and r["gold_mapped"] == lab for r in records)
        fp = sum(r["pred"] == lab and r["gold_mapped"] != lab for r in records)
        fn = sum(r["pred"] != lab and r["gold_mapped"] == lab for r in records)
        p = tp / (tp + fp) if tp + fp else 0.0
        rcl = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rcl / (p + rcl) if p + rcl else 0.0
        out[lab] = {"precision": p, "recall": rcl, "f1": f1, "support": tp + fn}
    return out


def _accuracy(records):
    return sum(r["correct"] for r in records) / len(records) if records else float("nan")


def _macro_f1(records):
    prf = _prf(records)
    present = [lab for lab in PRED_LABELS if prf[lab]["support"] > 0]
    return sum(prf[l]["f1"] for l in present) / len(present) if present else float("nan")


def _bootstrap_ci(records, fn, reps=2000, seed=0) -> Tuple[float, float]:
    if len(records) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = sorted(fn([rng.choice(records) for _ in records]) for _ in range(reps))
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def add_derived_flags(r: dict) -> dict:
    """Recompute correctness and leakage flags at score time."""
    if r.get("gold_label") in LABEL_MAP:
        r["gold_mapped"] = LABEL_MAP[r["gold_label"]]
        r["correct"] = r.get("pred") == r["gold_mapped"]
    urls = list((r.get("trace") or {}).get("retrieved_urls") or []) + list(r.get("sources") or [])
    r["factcheck_page_retrieved"] = bool(r.get("leak_factcheck_site")) or any(
        FACTCHECK_URL_RE.search(u) for u in urls
    )
    r["cites_factchecker"] = bool(FACTCHECKER_NAMES_RE.search(r.get("justification") or ""))
    return r


def _cost_of(r: dict) -> float:
    return r["cost_usd"] if isinstance(r.get("cost_usd"), (int, float)) else record_cost(r)


def condition_metrics(records: List[dict]) -> dict:
    ok = [add_derived_flags(r) for r in records if not r.get("error")]
    fc_page = [r for r in ok if r["factcheck_page_retrieved"]]
    no_fc_page = [r for r in ok if not r["factcheck_page_retrieved"]]
    answered = [r for r in ok if r["pred"] != "unverifiable"]
    clean = [r for r in ok if not r.get("leak_exact")]
    acc_ci = _bootstrap_ci(ok, _accuracy)
    f1_ci = _bootstrap_ci(ok, _macro_f1)

    confusion = {g: {p: 0 for p in PRED_LABELS} for g in GOLD_LABELS}
    for r in ok:
        confusion[r["gold_label"]][r["pred"]] += 1

    by_conf = {}
    for c in ("high", "medium", "low"):
        sub = [r for r in ok if r.get("confidence") == c]
        by_conf[c] = {"n": len(sub), "accuracy": _accuracy(sub) if sub else None}

    def mean(key):
        vals = [r["trace"][key] for r in ok if r.get("trace")]
        return sum(vals) / len(vals) if vals else 0.0

    versions = sorted({r.get("prompt_version", "v1") for r in records})
    return {
        "prompt_versions": versions,
        "n_total": len(records),
        "n_scored": len(ok),
        "n_errors": len(records) - len(ok),
        "accuracy": _accuracy(ok),
        "accuracy_ci95": acc_ci,
        "macro_f1": _macro_f1(ok),
        "macro_f1_ci95": f1_ci,
        "per_class": _prf(ok),
        "confusion": confusion,
        "coverage": len(answered) / len(ok) if ok else float("nan"),
        "selective_accuracy": _accuracy(answered),
        "by_confidence": by_conf,
        "leak_exact_rate": sum(bool(r.get("leak_exact")) for r in ok) / len(ok) if ok else 0.0,
        "leak_factcheck_site_rate": sum(bool(r.get("leak_factcheck_site")) for r in ok) / len(ok) if ok else 0.0,
        "factcheck_page_rate": len(fc_page) / len(ok) if ok else 0.0,
        "accuracy_with_factcheck_page": _accuracy(fc_page),
        "accuracy_without_factcheck_page": _accuracy(no_fc_page),
        "n_without_factcheck_page": len(no_fc_page),
        "cites_factchecker_rate": sum(r["cites_factchecker"] for r in ok) / len(ok) if ok else 0.0,
        "n_without_exact_leak": len(clean),
        "accuracy_without_exact_leak": _accuracy(clean),
        "macro_f1_without_exact_leak": _macro_f1(clean),
        "followup_rate": sum(bool(r["trace"].get("followup_used")) for r in ok) / len(ok) if ok else 0.0,
        "mean_input_tokens": mean("input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_web_searches": mean("web_search_requests"),
        "mean_seconds": sum(r["elapsed_s"] for r in ok) / len(ok) if ok else 0.0,
        "mean_cost_usd": sum(_cost_of(r) for r in ok) / len(ok) if ok else 0.0,
        "total_cost_usd": sum(_cost_of(r) for r in records),
        "batch_share": sum(bool(r.get("batch")) for r in ok) / len(ok) if ok else 0.0,
        "max_searches": sorted({r.get("max_searches", 5) for r in ok}),
    }


def paired_comparisons(by_cond: Dict[str, List[dict]]) -> List[dict]:
    rows = []
    conds = [c for c in CONDITIONS if c in by_cond] + [c for c in by_cond if c not in CONDITIONS]
    for i, a in enumerate(conds):
        for b in conds[i + 1 :]:
            ra = {r["id"]: r for r in by_cond[a] if not r.get("error")}
            rb = {r["id"]: r for r in by_cond[b] if not r.get("error")}
            common = sorted(set(ra) & set(rb))
            if not common:
                continue
            only_a = sum(ra[k]["correct"] and not rb[k]["correct"] for k in common)
            only_b = sum(rb[k]["correct"] and not ra[k]["correct"] for k in common)
            rows.append(
                {
                    "a": a, "b": b, "n_common": len(common),
                    "acc_a": _accuracy([ra[k] for k in common]),
                    "acc_b": _accuracy([rb[k] for k in common]),
                    "only_a_correct": only_a, "only_b_correct": only_b,
                    "mcnemar_p": _mcnemar_exact(only_a, only_b),
                }
            )
    return rows


# Report


def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _ci(ci):
    lo, hi = ci
    return "n/a" if math.isnan(lo) else f"{100 * lo:.1f} to {100 * hi:.1f}"


def write_markdown(metrics: Dict[str, dict], pairs: List[dict], path: str) -> None:
    L = ["# AVeriTeC evaluation (check 1: claim verification)", ""]
    L.append(f"Generated {dt.datetime.now().strftime('%d %B %Y, %H:%M')} with model `{fc.MODEL}`. "
             "Verdict label only; not comparable with the official AVeriTeC evidence-weighted score.")
    mixed = {c: m["prompt_versions"] for c, m in metrics.items() if len(m["prompt_versions"]) > 1}
    if mixed:
        L += ["", f"**Warning:** these conditions mix results from different prompt versions: {mixed}. "
              "Move the older results out of the results folder and score again."]
    L += ["", "Prompt versions: " + "; ".join(f"{c}: {', '.join(m['prompt_versions'])}" for c, m in metrics.items())]
    L += ["", "## Headline results", "",
          "| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Selective accuracy | Own fact-check retrieved |",
          "|---|---|---|---|---|---|---|"]
    for c, m in metrics.items():
        L.append(f"| {c} | {m['n_scored']} | {_pct(m['accuracy'])} ({_ci(m['accuracy_ci95'])}) | "
                 f"{_pct(m['macro_f1'])} ({_ci(m['macro_f1_ci95'])}) | {_pct(m['coverage'])} | "
                 f"{_pct(m['selective_accuracy'])} | {_pct(m['leak_exact_rate'])} |")
    L += ["", "Coverage is the share of claims given a definite verdict (true, false or misleading); "
          "selective accuracy is accuracy on those claims alone.", ""]

    L += ["## Per-class results", ""]
    for c, m in metrics.items():
        L += [f"**{c}**", "", "| Predicted class | Precision | Recall | F1 | Gold support |", "|---|---|---|---|---|"]
        for lab, v in m["per_class"].items():
            L.append(f"| {lab} | {_pct(v['precision'])} | {_pct(v['recall'])} | {_pct(v['f1'])} | {v['support']} |")
        L.append("")

    L += ["## Confusion matrices (rows: AVeriTeC gold label, columns: prediction)", ""]
    for c, m in metrics.items():
        L += [f"**{c}**", "", "| Gold | " + " | ".join(PRED_LABELS) + " |", "|---" * (len(PRED_LABELS) + 1) + "|"]
        for g in GOLD_LABELS:
            L.append(f"| {g} | " + " | ".join(str(m["confusion"][g][p]) for p in PRED_LABELS) + " |")
        L.append("")

    L += ["## Accuracy by stated confidence", "", "| Condition | High | Medium | Low |", "|---|---|---|---|"]
    for c, m in metrics.items():
        cells = [f"{_pct(m['by_confidence'][k]['accuracy'])} (n={m['by_confidence'][k]['n']})" for k in ("high", "medium", "low")]
        L.append(f"| {c} | " + " | ".join(cells) + " |")

    L += ["", "## Leakage", "",
          "| Condition | Own fact-check retrieved | Any fact-check page retrieved | Justification cites a fact-checker "
          "| Accuracy with a fact-check page | Accuracy without (n) |",
          "|---|---|---|---|---|---|"]
    for c, m in metrics.items():
        L.append(f"| {c} | {_pct(m['leak_exact_rate'])} | {_pct(m['factcheck_page_rate'])} | "
                 f"{_pct(m['cites_factchecker_rate'])} | {_pct(m['accuracy_with_factcheck_page'])} | "
                 f"{_pct(m['accuracy_without_factcheck_page'])} ({m['n_without_factcheck_page']}) |")
    L += ["", "A fact-check page is any retrieved URL from a blocked site or whose address marks it as a "
          "fact-check (for example cnn.com/.../biden-fracking-fact-check). Citing a fact-checker in the "
          "justification without retrieving its page points to knowledge from training or an indirect "
          "source, which blocking cannot prevent."]

    if pairs:
        L += ["", "## Paired comparison (same claims, exact McNemar test)", "",
              "| A | B | Claims | Accuracy A | Accuracy B | Only A right | Only B right | p |", "|---|---|---|---|---|---|---|---|"]
        for p in pairs:
            L.append(f"| {p['a']} | {p['b']} | {p['n_common']} | {_pct(p['acc_a'])} | {_pct(p['acc_b'])} | "
                     f"{p['only_a_correct']} | {p['only_b_correct']} | {p['mcnemar_p']:.4f} |")

    L += ["", "## Cost and time per claim", "",
          "| Condition | Input tokens | Output tokens | Web searches (cap) | Seconds | Batch API | Cost per claim (USD) | Total spent (USD) | Follow-up needed |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c, m in metrics.items():
        cap = ", ".join(str(x) for x in m["max_searches"])
        L.append(f"| {c} | {m['mean_input_tokens']:.0f} | {m['mean_output_tokens']:.0f} | "
                 f"{m['mean_web_searches']:.1f} ({cap}) | {m['mean_seconds']:.1f} | {_pct(m['batch_share'])} | "
                 f"{m['mean_cost_usd']:.4f} | {m['total_cost_usd']:.2f} | {_pct(m['followup_rate'])} |")
    L += ["", "Costs are estimates from token and search counts at the list prices in eval_averitec.py "
          "(checked 28 September 2026); the Anthropic Console shows the exact amount billed."]
    errs = {c: m["n_errors"] for c, m in metrics.items() if m["n_errors"]}
    if errs:
        L += ["", f"Claims excluded because of API or runtime errors: {errs}. Re-run with --retry-errors."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def plot_confusion(condition: str, confusion: dict, path: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return False

    # shade = share of the gold row
    ramp = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ramp)
    counts = [[confusion[g][p] for p in PRED_LABELS] for g in GOLD_LABELS]
    shares = [[v / sum(row) if sum(row) else 0.0 for v in row] for row in counts]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    ax.imshow(shares, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(counts):
        for j, v in enumerate(row):
            s = shares[i][j]
            ax.text(j, i, f"{v}\n{100 * s:.0f}%", ha="center", va="center", fontsize=9,
                    color="#ffffff" if s > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(PRED_LABELS)), ["Likely true", "Likely false", "Misleading", "Unverifiable"], fontsize=9)
    ax.set_yticks(range(len(GOLD_LABELS)),
                  ["Supported", "Refuted", "Not enough evidence", "Conflicting / cherrypicking"], fontsize=9)
    ax.set_xlabel("Pipeline verdict", fontsize=9, color="#52514e")
    ax.set_ylabel("AVeriTeC gold label", fontsize=9, color="#52514e")
    ax.set_title(f"Claim verification, {condition.replace('_', ' ')}\n(cell shade = share of gold row)",
                 fontsize=10, color="#0b0b0b")
    ax.tick_params(length=0, colors="#52514e")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(PRED_LABELS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(GOLD_LABELS))], minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def score(args) -> None:
    files = args.files or sorted(
        os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR)
        if f.startswith("averitec_") and f.endswith(".jsonl")
    ) if (args.files or os.path.isdir(RESULTS_DIR)) else []
    if not files:
        sys.exit("No results found. Run `python eval_averitec.py run ...` first.")

    by_cond: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prev = by_cond[rec["condition"]].get(rec["id"])
                # keep the latest successful attempt for each claim
                if prev is None or prev.get("error") or not rec.get("error"):
                    by_cond[rec["condition"]][rec["id"]] = rec
    by_cond_list = {c: sorted(d.values(), key=lambda r: r["id"]) for c, d in by_cond.items()}
    ordered = [c for c in CONDITIONS if c in by_cond_list] + [c for c in by_cond_list if c not in CONDITIONS]
    metrics = {c: condition_metrics(by_cond_list[c]) for c in ordered}
    pairs = paired_comparisons(by_cond_list)

    os.makedirs(args.out, exist_ok=True)
    write_markdown(metrics, pairs, os.path.join(args.out, "summary.md"))
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"model": fc.MODEL, "conditions": metrics, "paired": pairs}, f, indent=2)

    cols = ["condition", "id", "gold_label", "gold_mapped", "pred", "confidence", "correct",
            "prompt_version", "leak_exact", "factcheck_page_retrieved", "cites_factchecker", "elapsed_s", "error", "claim", "justification", "sources"]
    with open(os.path.join(args.out, "per_claim.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in ordered:
            for r in by_cond_list[c]:
                w.writerow({**r, "sources": " ".join(r.get("sources") or [])})

    plotted = [plot_confusion(c, metrics[c]["confusion"], os.path.join(args.out, f"confusion_{c}.png")) for c in ordered]

    print(f"\n{'Condition':<16}{'n':>5}{'Accuracy':>11}{'Macro F1':>11}{'Coverage':>11}{'Leak':>8}")
    for c in ordered:
        m = metrics[c]
        if len(m["prompt_versions"]) > 1:
            print(f"WARNING: {c} mixes prompt versions {m['prompt_versions']}; see summary.md")
        print(f"{c:<16}{m['n_scored']:>5}{_pct(m['accuracy']):>11}{_pct(m['macro_f1']):>11}"
              f"{_pct(m['coverage']):>11}{_pct(m['leak_exact_rate']):>8}")
    print(f"\nReport written to {args.out}{'' if all(plotted) else ' (install matplotlib for the figures)'}")


# CLI


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate factcheck.py claim verification on AVeriTeC.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Download the AVeriTeC development set.")
    d.add_argument("--data", default=DEFAULT_DATA)

    r = sub.add_parser("run", help="Run one condition over a sample of claims.")
    r.add_argument("--condition", choices=CONDITIONS, default="search_blocked")
    r.add_argument("--batch", action="store_true",
                   help="Use the Batch API: half the token price, results within 24 hours (usually under 1).")
    r.add_argument("--max-searches", type=int, default=3,
                   help="Web searches allowed per claim (default 3). Search results are the main cost.")
    r.add_argument("--budget", type=float, default=None,
                   help="Stop once about this many US dollars have been spent (not with --batch).")
    r.add_argument("--dry-run", action="store_true", help="Show the claims and cost estimate, then stop.")
    r.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt.")
    r.add_argument("--tag", default="", help="Suffix for the results file and condition name, e.g. haiku.")
    r.add_argument("--only", nargs="*", default=[],
                   help="Only use claims that appear in these results files (e.g. for cost tests on dev claims).")
    r.add_argument("--data", default=DEFAULT_DATA)
    r.add_argument("--sample", type=int, default=None, help="Stratified sample size (default: all 500).")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--workers", type=int, default=3, help="Claims checked in parallel.")
    r.add_argument("--no-context", action="store_true", help="Send the claim without speaker/date metadata.")
    r.add_argument("--retry-errors", action="store_true", help="Re-run claims that errored last time.")
    r.add_argument("--exclude", nargs="*", default=[],
                   help="Results files whose claims should be left out (for example the pilot).")

    s = sub.add_parser("score", help="Compute metrics and write the report.")
    s.add_argument("files", nargs="*", help="Results files (default: every results/averitec_*.jsonl).")
    s.add_argument("--out", default=os.path.join(RESULTS_DIR, "report"))

    args = ap.parse_args(argv)
    {"download": lambda: download(args.data), "run": lambda: run(args), "score": lambda: score(args)}[args.cmd]()


if __name__ == "__main__":
    main()
