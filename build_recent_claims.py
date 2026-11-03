"""Builds the post-cutoff claim set from the Google Fact Check Tools API in the
same format as the AVeriTeC file. Needs GOOGLE_FACTCHECK_API_KEY in .env.

    python build_recent_claims.py test
    python build_recent_claims.py fetch
    python eval_averitec.py run --data data/recent_claims_sample.json --condition search_blocked --batch --tag recent --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from dotenv import load_dotenv

import eval_averitec as ev

load_dotenv()
API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
OUT = os.path.join("data", "recent_claims.json")
SUMMARY = os.path.join("data", "recent_claims_summary.csv")
SAMPLE = os.path.join("data", "recent_claims_sample.json")

# Fact-checkers queried (all covered by the blocklist).
PUBLISHERS = [
    "politifact.com", "factcheck.org", "snopes.com", "leadstories.com", "checkyourfact.com",
    "factcheck.afp.com", "fullfact.org", "aap.com.au", "africacheck.org", "boomlive.in",
    "factly.in", "newsmeter.in", "altnews.in", "vishvasnews.com", "dubawa.org", "misbar.com",
    "healthfeedback.org", "sciencefeedback.co", "usatoday.com", "reuters.com", "apnews.com",
    "abc.net.au", "rappler.com", "verafiles.org", "truthorfiction.com", "polygraph.info",
]

# Rating text -> AVeriTeC label, first match wins.
RATING_RULES = [
    (r"\bsatire\b|\bparody\b|\bnot (a )?claim\b|\bopinion\b|\bresearch in progress\b|\bprebunk\b|\bjoke\b", None),
    # a rating that starts with a verdict word takes that verdict
    (r"^\W*(this is )?(a )?(false|fake|hoax|incorrect|wrong|fabricated|untrue|made[- ]up)\b", "Refuted"),
    (r"^\W*(this is )?(true|correct|accurate)\b", "Supported"),
    (r"\bmostly (true|correct|accurate)\b|\blargely (true|accurate|correct)\b", "Supported"),
    (r"\bmostly false\b|\blargely (false|inaccurate)\b", "Refuted"),
    (r"half[- ]true|\bmixture\b|\bmixed\b|\bmisleading\b|missing context|lacks context|needs context|"
     r"\bcherry|\bexaggerat|partly (false|true)|partially (false|true|correct)|half[- ]truth|"
     r"\bdistort|\bspin\b|out of context|\bimprecise\b|\boverstat|\bunderstat|not the whole story", "Conflicting Evidence/Cherrypicking"),
    (r"\bunproven\b|\bunverified\b|\bunsupported\b|\bno evidence\b|lacks evidence|\bunsubstantiated\b|"
     r"\bcannot be verified\b|\bnot proven\b|\binsufficient evidence\b", "Not Enough Evidence"),
    (r"\bfalse\b|\bfake\b|pants on fire|\bincorrect\b|\binaccurate\b|\bfabricat|\bhoax\b|\bnot true\b|"
     r"\buntrue\b|\bwrong\b|\bbaseless\b|\bscam\b|four pinocchios|\bbogus\b|\bdebunked\b|\bmyth\b|"
     r"\bmisattributed\b|\bfalsely\b|\bno,|\bmade[- ]up\b|\bnever said\b|\bdidn'?t say\b|\bdid not say\b|"
     r"\bdidn'?t happen\b|\bno such\b|incorrect attribution|not (a )?genuine|isn'?t genuine|isn'?t true|\bnot real\b", "Refuted"),
    (r"^\s*(true|correct|accurate|yes)\b|\bis true\b|\bis correct\b|\bis accurate\b|"
     r"\blegit\b|\bcorrect attribution\b", "Supported"),
]
MEDIA_RE = re.compile(r"\b(video|videos|photo|photos|photograph|image|images|picture|pictures|clip|footage|"
                      r"screenshot|meme|graphic)\b", re.I)


# Sentence ratings (Full Fact, AAP) are kept only if they start with a verdict word.
VERDICT_OPENING = re.compile(r"^\W*(this is |this isn'?t |that'?s |this claim is )?(a )?(false|true|correct|incorrect|"
                             r"misleading|wrong|fake|hoax|mostly|half|partly|untrue|not true|fabricated|made[- ]up)\b")


def map_rating(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None
    if len(t) > 45 and not VERDICT_OPENING.match(t):
        return None
    for pattern, label in RATING_RULES:
        if re.search(pattern, t):
            return label
    return None


def choose(pool: List[dict], target: int, seed: int) -> List[dict]:
    """Pick the evaluation sample, weighted towards the rarer labels."""
    import random

    rng = random.Random(seed)
    by: Dict[str, List[dict]] = defaultdict(list)
    for x in pool:
        by[x["label"]].append(x)
    for lst in by.values():
        rng.shuffle(lst)
    quota = {"Supported": min(len(by["Supported"]), target // 4),
             "Not Enough Evidence": min(len(by["Not Enough Evidence"]), target // 4)}
    quota["Conflicting Evidence/Cherrypicking"] = min(len(by["Conflicting Evidence/Cherrypicking"]), int(target * 0.3))
    quota["Refuted"] = min(len(by["Refuted"]), target - sum(quota.values()))
    out = [x for lab, n in quota.items() for x in by[lab][:n]]
    return sorted(out, key=lambda x: (x["review_date"] or "", x["fact_checking_article"]))


def _day(s: Optional[str]) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat((s or "")[:10])
    except ValueError:
        return None


def _get(params: dict) -> dict:
    import requests

    key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if not key:
        sys.exit("GOOGLE_FACTCHECK_API_KEY is not set. Add it to your .env file (see the top of this script).")
    for attempt in range(4):
        r = requests.get(API, params={**params, "key": key}, timeout=30)
        if r.status_code in (429, 500, 503):
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code >= 400:
            sys.exit(f"Fact Check Tools API error {r.status_code}: {r.text[:300]}")
        return r.json()
    sys.exit("Fact Check Tools API kept refusing requests; try again later.")


def test(args) -> None:
    """Try a few query shapes and report what the API returns."""
    variants = [
        ("publisher only", {"reviewPublisherSiteFilter": "politifact.com"}),
        ("publisher, English", {"reviewPublisherSiteFilter": "politifact.com", "languageCode": "en"}),
        ("publisher, last 200 days", {"reviewPublisherSiteFilter": "politifact.com", "maxAgeDays": 200}),
        ("keyword only", {"query": "election"}),
        ("keyword, last 200 days", {"query": "election", "maxAgeDays": 200}),
        ("AFP, last 200 days", {"reviewPublisherSiteFilter": "factcheck.afp.com", "maxAgeDays": 200}),
    ]
    sample = None
    for name, params in variants:
        data = _get({**params, "pageSize": 50})
        claims = data.get("claims", [])
        dates = sorted((rv.get("reviewDate") or "")[:10] for c in claims for rv in c.get("claimReview", []))
        print(f"  {name:<26} {len(claims):>3} claims   newest review {dates[-1] if dates else '-'}")
        if claims and sample is None:
            sample = claims[0]
    if sample:
        print("\nOne claim as returned:")
        print(json.dumps(sample, indent=1, ensure_ascii=False)[:1500])
    else:
        print("\nNo variant returned anything. Check that the key is restricted to the Fact Check Tools API "
              "and that the API is enabled for the project the key belongs to.")


def fetch(args) -> None:
    since = dt.date.fromisoformat(args.since)
    max_age = (dt.date.today() - since).days + 1
    raw: List[dict] = []
    for pub in args.publishers:
        token, got = None, 0
        while True:
            params = {"reviewPublisherSiteFilter": pub, "languageCode": "en", "pageSize": 100,
                      "maxAgeDays": max_age}
            if token:
                params["pageToken"] = token
            data = _get(params)
            for c in data.get("claims", []):
                for rv in c.get("claimReview", []):
                    raw.append({"claim": (c.get("text") or "").strip(), "claimant": c.get("claimant"),
                                "claim_date": c.get("claimDate"), "review_date": rv.get("reviewDate"),
                                "url": rv.get("url", ""), "publisher": (rv.get("publisher") or {}).get("site") or pub,
                                "rating": rv.get("textualRating", ""), "language": rv.get("languageCode", ""),
                                "queried": pub})
                    got += 1
            token = data.get("nextPageToken")
            if not token or got >= args.max_per_query:
                break
            time.sleep(0.3)
        print(f"  {pub:<22} {got} reviews")

    reasons, unmapped = Counter(), Counter()
    kept, seen_text, seen_url, per_pub = [], set(), set(), Counter()
    for r in raw:
        review_day, claim_day = _day(r["review_date"]), _day(r["claim_date"])
        text_key = re.sub(r"\W+", " ", r["claim"].lower()).strip()
        label = map_rating(r["rating"])
        if r["language"] and not r["language"].startswith("en"):
            reasons["not English"] += 1
        elif not r["claim"] or len(r["claim"]) < 15:
            reasons["no usable claim text"] += 1
        elif not review_day or review_day < since or (claim_day and claim_day < since):
            reasons[f"made or reviewed before {since}"] += 1
        elif not any(ev._matches_block(r["url"], e) for e in ev.BLOCKED_DOMAINS):
            reasons["review address not covered by the blocklist"] += 1
        elif MEDIA_RE.search(r["claim"]):
            reasons["about what a photo or video shows"] += 1
        elif label is None:
            reasons["rating does not map clearly"] += 1
            unmapped[r["rating"].strip().lower()[:60]] += 1
        elif text_key in seen_text or r["url"] in seen_url:
            reasons["duplicate"] += 1
        elif label in ("Refuted", "Conflicting Evidence/Cherrypicking") and per_pub[r["publisher"]] >= args.per_publisher:
            reasons["publisher cap reached"] += 1
        else:
            seen_text.add(text_key)
            seen_url.add(r["url"])
            per_pub[r["publisher"]] += 1
            day = claim_day or review_day
            kept.append({
                "claim": r["claim"], "label": label, "claim_date": day.strftime("%d-%m-%Y"),
                "speaker": r["claimant"] or None, "original_claim_url": None,
                "fact_checking_article": r["url"], "reporting_source": None, "location_ISO_code": None,
                "review_publisher": r["publisher"], "textual_rating": r["rating"],
                "review_date": r["review_date"], "date_source": "claim" if claim_day else "review",
            })

    kept.sort(key=lambda x: (x["review_date"] or "", x["fact_checking_article"]))
    chosen = choose(kept, args.target, args.seed)
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    with open(SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "item", "count"])
        for k, v in reasons.most_common():
            w.writerow(["dropped", k, v])
        for k, v in Counter(x["label"] for x in kept).most_common():
            w.writerow(["kept_label", k, v])
        for k, v in per_pub.most_common():
            w.writerow(["kept_publisher", k, v])
        for k, v in Counter((x["textual_rating"].strip().lower(), x["label"]) for x in kept).most_common():
            w.writerow(["rating_mapping", f"{k[0]} -> {k[1]}", v])
        for k, v in unmapped.most_common():
            w.writerow(["unmapped_rating", k, v])

    print(f"\n{len(raw)} reviews fetched; {len(kept)} claims kept and written to {OUT}.")
    print("Dropped:", dict(reasons.most_common()))
    print("Labels:", dict(Counter(x["label"] for x in kept).most_common()))
    print("Publishers:", dict(per_pub.most_common()))
    if unmapped:
        print("Most common ratings that were dropped as unclear:", [k for k, _ in unmapped.most_common(10)])
    with open(SAMPLE, "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=1)
    print(f"Evaluation sample of {len(chosen)} written to {SAMPLE}: {dict(Counter(x['label'] for x in chosen).most_common())}")
    print(f"Full breakdown, including every rating and its label, in {SUMMARY}.")
    if len(kept) < args.min_claims:
        print(f"Fewer than {args.min_claims} claims; try an earlier --since (not before 2026-02-01) or a higher --per-publisher.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build recent fact-checked claims in AVeriTeC format (free).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test", help="Print one raw API reply to check the key works.")
    f = sub.add_parser("fetch", help="Fetch, filter and write data/recent_claims.json.")
    f.add_argument("--since", default="2026-04-01", help="Earliest claim and review date (ISO).")
    f.add_argument("--publishers", nargs="*", default=PUBLISHERS)
    f.add_argument("--per-publisher", type=int, default=40, help="Most claims kept from one fact-checker.")
    f.add_argument("--max-per-query", type=int, default=400, help="Most reviews fetched per fact-checker.")
    f.add_argument("--min-claims", type=int, default=150)
    f.add_argument("--target", type=int, default=150, help="Size of the evaluation sample.")
    f.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    {"test": test, "fetch": fetch}[args.cmd](args)


if __name__ == "__main__":
    main()
