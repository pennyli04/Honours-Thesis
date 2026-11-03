"""Stage 3 (C2PA) evaluation on a controlled set signed with the c2pa-python
test certificate.

    python eval_c2pa.py make-set
    python eval_c2pa.py check
    python eval_c2pa.py run --condition rule
    python eval_c2pa.py run --condition model --dry-run
    python eval_c2pa.py run --condition model
    python eval_c2pa.py score
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import shutil
import sys
import time
from collections import Counter

import eval_averitec as ev
import eval_common as ec
import factcheck as fc

SET_DIR = os.path.join("data", "c2pa_set")
MANIFEST = os.path.join(SET_DIR, "manifest.csv")
CERT_DIR = "c2pa_test_certs"
PREFIX = "c2pa_"
CAPTION = "An unedited photograph taken by a staff photographer."
FIELDS = ["id", "file_path", "condition", "gold", "caption", "expected_manifest", "expected_failures", "notes"]

CONDITION_GOLD = {
    "signed_capture": "unverifiable",
    "ai_generated": "content_manipulation",
    "tampered": "content_manipulation",
    "stripped": "unverifiable",
    "unsigned": "unverifiable",
    "signed_trusted": "authentic_truthful",
}
GOLD = ["content_manipulation", "authentic_truthful", "unverifiable"]
PRED = ["content_manipulation", "miscontextualised", "authentic_truthful", "unverifiable"]
RUN_CONDITIONS = ("rule", "model")
DEFAULT_USAGE = {"input_tokens": 5500, "output_tokens": 700}

DST = "http://cv.iptc.org/newscodes/digitalsourcetype/"
AI_SOURCE_TYPES = ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia",
                   "algorithmicMedia", "compositeSynthetic")


# building the set


def _signer():
    import c2pa
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec as ecc

    certs = open(os.path.join(CERT_DIR, "es256_certs.pem"), "rb").read().decode()
    key = serialization.load_pem_private_key(open(os.path.join(CERT_DIR, "es256_private.key"), "rb").read(), None)
    return c2pa.Signer.from_callback(lambda data: key.sign(data, ecc.ECDSA(hashes.SHA256())),
                                     c2pa.C2paSigningAlg.ES256, certs, None)


def _sign(src: str, dest: str, source_type: str) -> None:
    import c2pa

    manifest = {
        "claim_generator_info": [{"name": "thesis_eval", "version": "1.0"}],
        "format": "image/jpeg",
        "title": "asset.jpg",
        "assertions": [{"label": "c2pa.actions",
                        "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": DST + source_type}]}}],
    }
    with _signer() as signer, c2pa.Builder(manifest) as builder:
        builder.sign_file(src, dest, signer)


def _tamper(path: str) -> None:
    """Change a few bytes of picture data so the recorded hash no longer matches."""
    b = bytearray(open(path, "rb").read())
    end = b.rfind(b"\xff\xd9")
    i, changed = max(2, end - 400), 0
    while changed < 16 and i < end - 2:
        if b[i] not in (0xFF, 0xFE, 0x00) and b[i - 1] != 0xFF:
            b[i] ^= 0x01
            changed += 1
        i += 7
    open(path, "wb").write(b)


def _synthetic_photos(n: int, out_dir: str, seed: int) -> list:
    """Placeholder pictures for when no photos are supplied."""
    from PIL import Image, ImageDraw

    rnd = random.Random(seed)
    paths = []
    for k in range(n):
        w, h = rnd.choice([(800, 600), (1024, 768), (640, 480)])
        im = Image.new("RGB", (w, h), tuple(rnd.randrange(40, 220) for _ in range(3)))
        d = ImageDraw.Draw(im)
        for _ in range(12):
            x0, y0 = rnd.randrange(w), rnd.randrange(h)
            box = [x0, y0, min(w, x0 + rnd.randrange(40, 300)), min(h, y0 + rnd.randrange(40, 300))]
            (d.ellipse if rnd.random() < 0.5 else d.rectangle)(box, fill=tuple(rnd.randrange(256) for _ in range(3)))
        p = os.path.join(out_dir, f"base_{k:03d}.jpg")
        im.save(p, "JPEG", quality=90)
        paths.append(p)
    return paths


def make_set(args) -> None:
    from PIL import Image

    try:
        import c2pa  # noqa: F401
        import cryptography  # noqa: F401
    except ImportError:
        sys.exit("Needs c2pa-python and cryptography: pip install c2pa-python cryptography")
    if os.path.exists(SET_DIR) and os.listdir(SET_DIR) and not args.force:
        sys.exit(f"{SET_DIR} already exists. Use --force to rebuild it (this replaces the files).")
    shutil.rmtree(SET_DIR, ignore_errors=True)
    work = os.path.join(SET_DIR, "_work")
    os.makedirs(work, exist_ok=True)

    conditions = ["signed_capture", "ai_generated", "tampered", "stripped", "unsigned"]
    n = args.per_condition * len(conditions)
    if args.photos:
        photos = sorted(os.path.join(args.photos, f) for f in os.listdir(args.photos)
                        if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not photos:
            sys.exit(f"No .jpg or .png files in {args.photos}")
        bases = []
        for k in range(n):  # reuse photos if there are fewer than needed
            src = photos[k % len(photos)]
            dest = os.path.join(work, f"base_{k:03d}.jpg")
            with Image.open(src) as im:
                im.convert("RGB").save(dest, "JPEG", quality=92)
            bases.append(dest)
    else:
        bases = _synthetic_photos(n, work, args.seed)

    plan = [c for c in conditions for _ in range(args.per_condition)]
    random.Random(args.seed).shuffle(plan)
    rows = []
    for k, (cond, base) in enumerate(zip(plan, bases), start=1):
        out = os.path.join(SET_DIR, f"asset_{k:03d}.jpg")
        expected_manifest, failures = "yes", "signingCredential.untrusted"
        if cond == "unsigned":
            shutil.copy(base, out)
            expected_manifest, failures = "no", ""
        else:
            _sign(base, out, "trainedAlgorithmicMedia" if cond == "ai_generated" else "digitalCapture")
            if cond == "tampered":
                _tamper(out)
                failures = "signingCredential.untrusted;assertion.dataHash.mismatch"
            elif cond == "stripped":
                with Image.open(out) as im:
                    im.convert("RGB").save(out + ".tmp.jpg", "JPEG", quality=92)
                os.replace(out + ".tmp.jpg", out)
                expected_manifest, failures = "no", ""
        rows.append({"id": f"asset_{k:03d}", "file_path": out.replace("\\", "/"), "condition": cond,
                     "gold": CONDITION_GOLD[cond], "caption": CAPTION,
                     "expected_manifest": expected_manifest, "expected_failures": failures,
                     "notes": "signed with the c2pa-python test certificate" if expected_manifest == "yes" or cond == "stripped" else ""})
    shutil.rmtree(work, ignore_errors=True)
    ec.write_csv(MANIFEST, rows, FIELDS)
    print(f"Built {len(rows)} files in {SET_DIR} ({dict(Counter(r['condition'] for r in rows))}).")
    print(f"Answers are in {MANIFEST}. To add trusted files, copy them into {SET_DIR} and add rows with "
          "condition signed_trusted, gold authentic_truthful, expected_manifest yes.")
    print("Next, a free check that the reader sees what was built: python eval_c2pa.py check")


# free checks


def _failure_codes(tool: dict) -> list:
    return [f.get("code") for f in ((tool.get("validation_results") or {}).get("failures") or []) if f.get("code")]


def check(args) -> None:
    rows = ec.read_csv(args.manifest)
    bad, out = 0, []
    for r in rows:
        tool = fc.check_c2pa_manifest(r["file_path"])
        present = "error" if "error" in tool else ("yes" if tool.get("manifest_present") else "no")
        codes = _failure_codes(tool)
        want = [c for c in (r.get("expected_failures") or "").split(";") if c]
        ok = present == (r.get("expected_manifest") or present) and all(c in codes for c in want)
        bad += not ok
        out.append({**r, "found_manifest": present, "found_state": tool.get("validation_state", ""),
                    "found_failures": ";".join(codes), "matches_expected": "yes" if ok else "NO",
                    "error": tool.get("error", "")})
    os.makedirs(ec.RESULTS_DIR, exist_ok=True)
    path = os.path.join(ec.RESULTS_DIR, "c2pa_check.csv")
    ec.write_csv(path, out, ["id", "condition", "expected_manifest", "found_manifest", "found_state",
                             "expected_failures", "found_failures", "matches_expected", "error"])
    print(f"{len(rows) - bad} of {len(rows)} files read exactly as expected. Details in {path}.")
    by = Counter((r["condition"], r["matches_expected"]) for r in out)
    for cond in sorted({r["condition"] for r in out}):
        print(f"  {cond:<16} {by[(cond, 'yes')]} as expected, {by[(cond, 'NO')]} not")
    if bad:
        print("Rows marked NO read differently from what was built; check them before running the model.")


def rule_category(tool: dict) -> str:
    if "error" in tool or not tool.get("manifest_present"):
        return "unverifiable"
    codes = _failure_codes(tool)
    serious = [c for c in codes if not c.startswith("signingCredential.untrusted") and ".ocsp" not in c]
    if serious or str(tool.get("validation_state", "")).lower() == "invalid":
        return "content_manipulation"
    for a in tool.get("actions") or []:
        if any(t in (a.get("digital_source_type") or "") for t in AI_SOURCE_TYPES):
            return "content_manipulation"
    return "authentic_truthful"


# running


def evaluate(row: dict, args) -> dict:
    trace = fc.new_trace()
    started = time.time()
    rec = {"id": row["id"], "condition": args.label, "base_condition": args.condition,
           "model": fc.MODEL if args.condition == "model" else "rule",
           "prompt_version": fc.PROVENANCE_PROMPT_VERSION, "file_condition": row.get("condition", ""),
           "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "gold": row["gold"], "error": None}
    try:
        if args.condition == "rule":
            tool = fc.check_c2pa_manifest(row["file_path"])
            trace["tool_results"] = [tool]
            rec.update(pred=rule_category(tool), confidence="", justification="rule on validation result")
        else:
            ext = os.path.splitext(row["file_path"])[1] or ".jpg"
            v = fc.verify_provenance(row["file_path"], caption=row.get("caption") or None,
                                     trace=trace, display_name=f"asset{ext}")
            rec.update(pred=v.category, verdict=v.verdict, confidence=v.confidence,
                       justification=v.justification, manifest_present=v.manifest_present,
                       validation_state=v.validation_state, issuer=v.issuer)
        rec["correct"] = rec["pred"] == rec["gold"]
    except Exception as exc:
        rec.update(pred=None, correct=None, error=f"{type(exc).__name__}: {exc}")
    rec["elapsed_s"] = round(time.time() - started, 2)
    rec["trace"] = trace
    rec["cost_usd"] = round(ec.token_cost(trace, fc.MODEL), 5)
    return rec


def run(args) -> None:
    args.label = args.condition + (f"_{args.tag}" if args.tag else "")
    rows = [r for r in ec.read_csv(args.manifest) if r.get("gold") in GOLD and r.get("file_path")]
    if not rows:
        sys.exit(f"No usable rows in {args.manifest}. Run `python eval_c2pa.py make-set` first.")
    for r in rows:
        r["_id"], r["label"] = r["id"], r.get("condition") or r["gold"]
    items = ec.sample(rows, args.sample, args.seed)
    path = os.path.join(ec.RESULTS_DIR, f"{PREFIX}{args.label}.jsonl")
    done = ec.done_ids(path, args.retry_errors)
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.label}: {len(items)} files ({dict(Counter(i['label'] for i in items))}), "
          f"{len(items) - len(todo)} already done, {len(todo)} to run.")
    if not todo:
        return
    if args.condition == "rule":
        per, basis = 0.0, "no API calls"
    else:
        usage = ec.observed_usage(PREFIX, args.label) or DEFAULT_USAGE
        p_in, p_out = ev.model_prices(fc.MODEL)
        per = (usage["input_tokens"] * p_in + usage["output_tokens"] * p_out) / 1e6
        basis = f"from {usage['n']} earlier files" if "n" in usage else "initial estimate"
    if not ec.confirm_cost(len(todo), per, basis, args):
        return
    if args.condition == "model":
        fc.get_client()
    writer = ec.Writer(path, len(todo), args.budget)
    ec.run_items(todo, lambda it: evaluate(it, args), args.workers if args.condition == "model" else 1, writer)
    writer.close()
    print(f"Done. {writer.errors} errors, about US${writer.spent:.2f} spent. Results in {path}")


# scoring


def score(args) -> None:
    files = args.files or ec.results_files(PREFIX)
    if not files:
        sys.exit("No C2PA results found. Run `python eval_c2pa.py run ...` first.")
    by_cond = ec.load_results(files)
    order = [c for c in RUN_CONDITIONS if c in by_cond] + [c for c in by_cond if c not in RUN_CONDITIONS]
    metrics = {c: ec.summarise(by_cond[c], GOLD, PRED) for c in order}
    os.makedirs(args.out, exist_ok=True)

    any_recs = [r for c in order for r in by_cond[c] if not r.get("error")]
    majority = Counter(r["gold"] for r in by_cond[order[0]] if not r.get("error")).most_common(1)[0]
    n0 = sum(1 for r in by_cond[order[0]] if not r.get("error"))
    L = ["# Stage 3 C2PA evaluation", "", f"Generated {dt.datetime.now():%d %B %Y, %H:%M}.", "",
         f"Always answering the most common correct category ({majority[0]}) would score "
         f"{ec.pct(majority[1] / n0)}; any useful result must beat that.", "",
         "| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Cost (USD) |", "|---|---|---|---|---|"]
    for c in order:
        m = metrics[c]
        L.append(f"| {c} | {m['n_scored']} | {ec.pct(m['accuracy'])} ({ec.ci(m['accuracy_ci95'])}) | "
                 f"{ec.pct(m['macro_f1'])} ({ec.ci(m['macro_f1_ci95'])}) | {m['total_cost_usd']:.2f} |")
    file_conds = sorted({r.get("file_condition") for r in any_recs}, key=lambda x: list(CONDITION_GOLD).index(x) if x in CONDITION_GOLD else 99)
    L += ["", "## Accuracy by type of file", "", "| File condition | Correct answer | " + " | ".join(order) + " |",
          "|---|---|" + "---|" * len(order)]
    for fcnd in file_conds:
        cells = []
        for c in order:
            xs = [r for r in by_cond[c] if not r.get("error") and r.get("file_condition") == fcnd]
            cells.append(f"{ec.pct(ec.accuracy(xs))} (n={len(xs)})" if xs else "n/a")
        L.append(f"| {fcnd} | {CONDITION_GOLD.get(fcnd, '')} | " + " | ".join(cells) + " |")
    pairs = ec.paired({c: by_cond[c] for c in order})
    if pairs:
        L += ["", "## Paired comparison (same files, exact McNemar test)", "",
              "| A | B | Files | Accuracy A | Accuracy B | Only A right | Only B right | p |", "|---|---|---|---|---|---|---|---|"]
        for p in pairs:
            L.append(f"| {p['a']} | {p['b']} | {p['n_common']} | {ec.pct(p['acc_a'])} | {ec.pct(p['acc_b'])} | "
                     f"{p['only_a_correct']} | {p['only_b_correct']} | {p['mcnemar_p']:.4f} |")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    rows = [r for c in order for r in by_cond[c]]
    ec.write_csv(os.path.join(args.out, "per_file.csv"), rows,
                 ["condition", "id", "file_condition", "gold", "pred", "verdict", "confidence", "correct",
                  "manifest_present", "validation_state", "issuer", "cost_usd", "error", "justification"])
    for c in order:
        ec.plot_confusion(metrics[c]["confusion"], GOLD, PRED, f"Stage 3, {c}", "Predicted category",
                          "Correct category", os.path.join(args.out, f"confusion_{c}.png"))
    print(f"\n{'Condition':<12}{'n':>5}{'Accuracy':>11}{'Macro F1':>11}{'Cost':>9}")
    for c in order:
        m = metrics[c]
        print(f"{c:<12}{m['n_scored']:>5}{ec.pct(m['accuracy']):>11}{ec.pct(m['macro_f1']):>11}{m['total_cost_usd']:>9.2f}")
    print(f"\nReport written to {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate stage 3 C2PA provenance checking.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-set", help="Build the controlled test set (free).")
    m.add_argument("--photos", default=None, help="Folder of your own .jpg/.png photos to sign (optional).")
    m.add_argument("--per-condition", type=int, default=12)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--force", action="store_true")
    c = sub.add_parser("check", help="Check the reader sees each file as built (free).")
    c.add_argument("--manifest", default=MANIFEST)
    r = sub.add_parser("run", help="Run the rule baseline or the model.")
    r.add_argument("--condition", choices=RUN_CONDITIONS, default="model")
    r.add_argument("--manifest", default=MANIFEST)
    r.add_argument("--sample", type=int, default=None)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--budget", type=float, default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--yes", action="store_true")
    r.add_argument("--retry-errors", action="store_true")
    r.add_argument("--tag", default="")
    s = sub.add_parser("score", help="Compute metrics and write the report.")
    s.add_argument("files", nargs="*")
    s.add_argument("--out", default=os.path.join(ec.RESULTS_DIR, "c2pa_report"))
    args = ap.parse_args(argv)
    {"make-set": make_set, "check": check, "run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
