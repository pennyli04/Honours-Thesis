"""Stage 2 (image) evaluation. Conditions: full, no_image, classifier_only.

    python eval_images.py prepare --manifest data/images.csv
    python eval_images.py run --condition classifier_only
    python eval_images.py run --condition full --sample 150 --dry-run
    python eval_images.py run --condition full --sample 150
    python eval_images.py run --condition no_image --sample 150
    python eval_images.py score
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from collections import Counter

import eval_averitec as ev
import eval_common as ec
import factcheck as fc

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
SMALL_DIR = os.path.join(DATA_DIR, "images_small")
PREPARED = os.path.join(DATA_DIR, "images_prepared.csv")
FIELDS = ["id", "label", "caption", "image_url", "image_path", "source", "fact_check_url", "notes"]
GOLD = ["content_manipulation", "miscontextualised", "authentic_truthful"]
PRED = GOLD + ["unverifiable"]
CONDITIONS = ("full", "no_image", "classifier_only")
PREFIX = "images_"

# Rough usage per image, for estimates before any run exists.
DEFAULT_USAGE = {
    "full": {"input_tokens": 9000, "output_tokens": 1000, "serpapi_searches": 1},
    "no_image": {"input_tokens": 5500, "output_tokens": 900, "serpapi_searches": 1},
}

# Classifier labels that count as manipulated.
FAKE_WORDS = ("fake", "deepfake", "synthetic", "generated", "manipulat", "artificial")


# template and preparation


def template(_args) -> None:
    rows = [
        {"id": "ammeba_0001", "label": "content_manipulation", "caption": "Caption or claim shared with the image",
         "image_url": "https://example.com/image1.jpg", "source": "AMMeBa",
         "fact_check_url": "https://www.snopes.com/fact-check/example/", "notes": "example row, delete"},
        {"id": "newsclip_0001", "label": "miscontextualised", "caption": "A caption taken from a different story",
         "image_url": "", "image_path": "data/images/newsclip_0001.jpg", "source": "NewsCLIPpings",
         "notes": "example row, delete"},
        {"id": "newsclip_0002", "label": "authentic_truthful", "caption": "The image's original caption",
         "image_url": "https://example.com/image2.jpg", "source": "NewsCLIPpings", "notes": "example row, delete"},
    ]
    path = os.path.join(DATA_DIR, "images_template.csv")
    ec.write_csv(path, rows, FIELDS)
    print(f"Wrote {path}. One row per image:\n"
          "  id              any unique name\n"
          f"  label           one of {', '.join(GOLD)}\n"
          "  caption         the claim published with the image (leave blank if the claim is printed on it)\n"
          "  image_url       a public link to the image (needed for reverse image search)\n"
          "  image_path      a local copy, if you have one (otherwise prepare downloads image_url)\n"
          "  source          AMMeBa, NewsCLIPpings, ...\n"
          "  fact_check_url  the fact check the image came from, if known (used to measure leakage)\n"
          "Save your filled-in copy as data/images.csv, then run: python eval_images.py prepare")


def _download(url: str, dest: str) -> str:
    import requests

    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (research evaluation)"})
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    return dest


def _shrink(src: str, dest: str, max_px: int) -> None:
    from PIL import Image

    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        im.save(dest, "JPEG", quality=90)


def prepare(args) -> None:
    """Download images, check they open and make the smaller copies. No API calls."""
    from PIL import Image

    rows = ec.read_csv(args.manifest)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(SMALL_DIR, exist_ok=True)
    out, status, ready = [], Counter(), Counter()
    for row in rows:
        row = dict(row)
        rid = row.get("id")
        if args.per_label and row.get("label") in GOLD and ready[row["label"]] >= args.per_label:
            row["status"] = "spare"
            status["spare"] += 1
            out.append(row)
            continue
        if not rid or row.get("label") not in GOLD:
            row["status"] = "bad_row"
            status["bad_row"] += 1
            out.append(row)
            continue
        path = row.get("image_path")
        try:
            if not path or not os.path.isfile(path):
                if not row.get("image_url"):
                    raise ValueError("no image_path or image_url")
                path = _download(row["image_url"], os.path.join(IMG_DIR, f"{rid}.img"))
                row["image_path"] = path
            with Image.open(path) as im:
                im.verify()
            small = os.path.join(SMALL_DIR, f"{rid}.jpg")
            _shrink(path, small, args.max_px)
            row["small_path"] = small
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}: {str(exc)[:80]}"
        status["ok" if row["status"] == "ok" else "failed"] += 1
        if row["status"] == "ok":
            ready[row["label"]] += 1
        out.append(row)
        print(f"  {rid:<28} {row['status'][:60]}", flush=True)
    ec.write_csv(args.out, out, FIELDS + ["group", "small_path", "status"])
    ok = [r for r in out if r.get("status") == "ok"]
    print(f"Prepared {len(ok)} of {len(rows)} images ({dict(status)}). Written to {args.out}.")
    print("Ready by label:", dict(Counter(r["label"] for r in ok)))
    print("Ready by source group:", dict(Counter(f"{r['label']}/{r.get('group') or r.get('source')}" for r in ok)))
    short = [lab for lab in GOLD if args.per_label and ready[lab] < args.per_label]
    if short:
        print(f"Fewer than {args.per_label} ready for: {', '.join(short)}. Build more candidates (a higher --oversample).")
    no_url = sum(1 for r in ok if not r.get("image_url"))
    if no_url:
        print(f"{no_url} ready images have no public URL, so reverse image search cannot run on them.")
    if len(ok) < len(rows):
        print("Report the number of images that could not be retrieved in your methods section.")


# running


def _classifier_category(result: dict) -> tuple:
    preds = result.get("predictions") or []
    if not preds:
        return "unverifiable", "", 0.0
    top = max(preds, key=lambda p: p.get("score", 0))
    label = str(top.get("label", ""))
    fake = any(w in label.lower() for w in FAKE_WORDS) and "real" not in label.lower()
    return ("content_manipulation" if fake else "authentic_truthful"), label, float(top.get("score", 0))


def serpapi_searches_left():
    """Searches left on the SerpApi plan, or None if it cannot be checked."""
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        return None
    try:
        import requests
        r = requests.get("https://serpapi.com/account.json", params={"api_key": key}, timeout=20)
        r.raise_for_status()
        d = r.json()
        for k in ("total_searches_left", "plan_searches_left"):
            if isinstance(d.get(k), int):
                return d[k]
    except Exception as exc:  # noqa: BLE001
        print(f"(Could not check remaining SerpApi searches: {exc})")
    return None


def evaluate(row: dict, args) -> dict:
    trace = fc.new_trace()
    started = time.time()
    rec = {
        "id": row["id"], "condition": args.label, "base_condition": args.condition,
        "model": fc.MODEL if args.condition != "classifier_only" else fc.MANIPULATION_MODEL_ID,
        "prompt_version": fc.IMAGE_PROMPT_VERSION, "source": row.get("source", ""), "group": row.get("group", ""),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gold": row["label"], "caption": row.get("caption", ""), "has_url": bool(row.get("image_url")),
        "blocked": not args.no_block, "error": None,
    }
    try:
        if args.condition == "classifier_only":
            result = fc.classify_manipulation(row["image_path"])
            if "error" in result:
                raise RuntimeError(result["error"])
            pred, label, score = _classifier_category(result)
            rec.update(pred=pred, confidence="", manipulation_label=label, manipulation_score=score,
                       justification="", sources=[])
        else:
            v = fc.verify_image(
                row["image_path"], image_url=row.get("image_url") or None, caption=row.get("caption") or None,
                trace=trace, blocked_domains=None if args.no_block else ev.BLOCKED_DOMAINS,
                send_image=args.condition == "full", model_image_path=row.get("small_path") or None,
            )
            # stop if reverse search fails (e.g. SerpApi quota used up)
            failed = [x["error"] for x in trace.get("tool_results", [])
                      if x.get("tool") == "reverse_image_search" and "error" in x
                      and "no public image URL" not in x["error"]]
            if failed:
                raise RuntimeError(f"SerpApi unavailable: {failed[0][:160]}")
            rec.update(pred=v.category, verdict=v.verdict, confidence=v.confidence,
                       justification=v.justification, sources=v.sources,
                       manipulation_label=v.manipulation_label, manipulation_score=v.manipulation_score)
        rec["correct"] = rec["pred"] == rec["gold"]
        urls = trace.get("retrieved_urls", []) + list(rec.get("sources") or [])
        fcu = row.get("fact_check_url")
        rec["leak_exact"] = bool(fcu) and any(ev._norm_url(u) == ev._norm_url(fcu) for u in urls)
        rec["factcheck_page_retrieved"] = any(ev.FACTCHECK_URL_RE.search(u) or fc._blocked(u, ev.BLOCKED_DOMAINS)
                                              for u in urls)
    except Exception as exc:
        rec.update(pred=None, correct=None, error=f"{type(exc).__name__}: {exc}")
    rec["elapsed_s"] = round(time.time() - started, 2)
    rec["trace"] = trace
    rec["cost_usd"] = round(ec.token_cost(trace, fc.MODEL), 5)
    return rec


def run(args) -> None:
    args.label = args.condition + (f"_{args.tag}" if args.tag else "")
    rows = [r for r in ec.read_csv(args.manifest) if r.get("status") == "ok" and r.get("label") in GOLD]
    if not rows:
        sys.exit(f"No ready images in {args.manifest}. Run `python eval_images.py prepare` first.")
    for r in rows:
        r["_id"] = r["id"]
    if args.exclude:
        gone = ec.read_ids(args.exclude)
        rows = [r for r in rows if r["id"] not in gone]
    items = ec.sample(rows, args.sample, args.seed)
    path = os.path.join(ec.RESULTS_DIR, f"{PREFIX}{args.label}.jsonl")
    done = ec.done_ids(path, args.retry_errors)
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.label}: {len(items)} images selected ({dict(Counter(i['label'] for i in items))}), "
          f"{len(items) - len(todo)} already done, {len(todo)} to run.")
    if not todo:
        return

    if args.condition != "classifier_only":
        left = serpapi_searches_left()
        if left is not None:
            print(f"SerpApi searches left this month: {left} (this run needs about {len(todo)}).")
            if left < len(todo):
                sys.exit("Not enough SerpApi searches left; wait for the monthly reset or run a smaller --sample.")
    if args.condition == "classifier_only":
        per, basis, note = 0.0, "runs locally", " No API calls; the first run downloads the classifier."
    else:
        usage = ec.observed_usage(PREFIX, args.label) or DEFAULT_USAGE[args.condition]
        p_in, p_out = ev.model_prices(fc.MODEL)
        per = (usage["input_tokens"] * p_in + usage["output_tokens"] * p_out) / 1e6
        basis = f"from {usage['n']} earlier images" if "n" in usage else "initial estimate for 1,000-pixel images"
        searches = round(usage.get("serpapi_searches", 1) * len(todo))
        note = (f" Plus about {searches} reverse image searches, free within SerpApi's 250 a month.")
        if not fc.SERPAPI_KEY:
            note += " SERPAPI_API_KEY is not set, so reverse image search will report an error to the model."
    if not ec.confirm_cost(len(todo), per, basis, args, note):
        return
    if args.condition != "classifier_only":
        fc.get_client()

    writer = ec.Writer(path, len(todo), args.budget)
    ec.run_items(todo, lambda it: evaluate(it, args), 1 if args.condition == "classifier_only" else args.workers, writer)
    writer.close()
    print(f"Done. {writer.errors} errors, about US${writer.spent:.2f} spent. Results in {path}")


# scoring


def score(args) -> None:
    files = args.files or ec.results_files(PREFIX)
    if not files:
        sys.exit("No image results found. Run `python eval_images.py run ...` first.")
    by_cond = ec.load_results(files)
    order = [c for c in CONDITIONS if c in by_cond] + [c for c in by_cond if c not in CONDITIONS]
    metrics = {c: ec.summarise(by_cond[c], GOLD, PRED) for c in order}
    pairs = ec.paired({c: by_cond[c] for c in order})
    os.makedirs(args.out, exist_ok=True)

    L = ["# Stage 2 image evaluation", "",
         f"Generated {dt.datetime.now():%d %B %Y, %H:%M}. Categories follow Dufour et al. (2024).", "",
         "| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Fact-check page returned | Cost (USD) | Reverse searches |",
         "|---|---|---|---|---|---|---|---|"]
    for c in order:
        m = metrics[c]
        ok = [r for r in by_cond[c] if not r.get("error")]
        leak = sum(bool(r.get("factcheck_page_retrieved")) for r in ok) / len(ok) if ok else 0
        L.append(f"| {c} | {m['n_scored']} | {ec.pct(m['accuracy'])} ({ec.ci(m['accuracy_ci95'])}) | "
                 f"{ec.pct(m['macro_f1'])} ({ec.ci(m['macro_f1_ci95'])}) | {ec.pct(m['coverage'])} | "
                 f"{ec.pct(leak)} | {m['total_cost_usd']:.2f} | {m['serpapi_searches']} |")
    L += ["", "## Accuracy by gold category", "", "| Condition | " + " | ".join(ec.CATEGORY_NAMES[g] for g in GOLD) + " |",
          "|---" * (len(GOLD) + 1) + "|"]
    for c in order:
        ok = [r for r in by_cond[c] if not r.get("error")]
        cells = []
        for g in GOLD:
            xs = [r for r in ok if r["gold"] == g]
            cells.append(f"{ec.pct(ec.accuracy(xs)) if xs else 'n/a'} (n={len(xs)})")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## Accuracy by source and type of item", "", "| Condition | Gold category | Source / type | n | Accuracy |",
          "|---|---|---|---|---|"]

    def key(r):
        return (r["gold"], f"{r.get('source') or '(none)'} / {r.get('group') or '-'}")

    for c in order:
        ok = [r for r in by_cond[c] if not r.get("error")]
        for (gold, src), n in sorted(Counter(key(r) for r in ok).items()):
            xs = [r for r in ok if key(r) == (gold, src)]
            L.append(f"| {c} | {gold} | {src} | {n} | {ec.pct(ec.accuracy(xs))} |")
    L += ["", "## Stated confidence (model conditions)", "", "| Condition | High | Medium | Low |", "|---|---|---|---|"]
    for c in order:
        bc = metrics[c]["by_confidence"]
        if any(bc[k]["n"] for k in bc):
            L.append(f"| {c} | " + " | ".join(f"{ec.pct(bc[k]['accuracy'])} (n={bc[k]['n']})" for k in ("high", "medium", "low")) + " |")
    if pairs:
        L += ["", "## Paired comparison (same images, exact McNemar test)", "",
              "| A | B | Images | Accuracy A | Accuracy B | Only A right | Only B right | p |", "|---|---|---|---|---|---|---|---|"]
        for p in pairs:
            L.append(f"| {p['a']} | {p['b']} | {p['n_common']} | {ec.pct(p['acc_a'])} | {ec.pct(p['acc_b'])} | "
                     f"{p['only_a_correct']} | {p['only_b_correct']} | {p['mcnemar_p']:.4f} |")
    L += ["", "The classifier-only condition can only answer content manipulation or authentic, so it cannot be right on miscontextualised images by design."]
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    rows = [dict(r, sources=" ".join(r.get("sources") or [])) for c in order for r in by_cond[c]]
    ec.write_csv(os.path.join(args.out, "per_image.csv"), rows,
                 ["condition", "id", "source", "gold", "pred", "verdict", "confidence", "correct",
                  "manipulation_label", "manipulation_score", "has_url", "leak_exact", "group",
                  "factcheck_page_retrieved", "cost_usd", "error", "caption", "justification", "sources"])
    for c in order:
        ec.plot_confusion(metrics[c]["confusion"], GOLD, PRED,
                          f"Stage 2, {c.replace('_', ' ')}", "Pipeline category", "Gold category",
                          os.path.join(args.out, f"confusion_{c}.png"))
    print(f"\n{'Condition':<18}{'n':>5}{'Accuracy':>11}{'Macro F1':>11}{'Cost':>9}")
    for c in order:
        m = metrics[c]
        print(f"{c:<18}{m['n_scored']:>5}{ec.pct(m['accuracy']):>11}{ec.pct(m['macro_f1']):>11}{m['total_cost_usd']:>9.2f}")
    print(f"\nReport written to {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate stage 2 image verification.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template", help="Write a CSV template to fill in.")
    p = sub.add_parser("prepare", help="Download and check images, make smaller copies (free).")
    p.add_argument("--manifest", default=os.path.join(DATA_DIR, "images.csv"))
    p.add_argument("--max-px", type=int, default=1000, help="Long side of the copy sent to the model.")
    p.add_argument("--per-label", type=int, default=None,
                   help="Stop once this many images per category have downloaded; the rest are kept as spares.")
    p.add_argument("--out", default=PREPARED, help="Where to write the prepared list (use a new name for a second set).")
    r = sub.add_parser("run", help="Run one condition.")
    r.add_argument("--condition", choices=CONDITIONS, default="full")
    r.add_argument("--manifest", default=PREPARED)
    r.add_argument("--sample", type=int, default=None)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--exclude", nargs="*", default=[])
    r.add_argument("--no-block", action="store_true", help="Do not withhold fact-checking sites from search results.")
    r.add_argument("--budget", type=float, default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--yes", action="store_true")
    r.add_argument("--retry-errors", action="store_true")
    r.add_argument("--tag", default="")
    s = sub.add_parser("score", help="Compute metrics and write the report.")
    s.add_argument("files", nargs="*")
    s.add_argument("--out", default=os.path.join(ec.RESULTS_DIR, "images_report"))
    args = ap.parse_args(argv)
    {"template": template, "prepare": prepare, "run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
