"""Builds the stage 2 image lists from AMMeBa and Wikimedia Commons.

    python build_image_set.py commons --test
    python build_image_set.py ammeba
    python build_image_set.py commons
    python build_image_set.py combine
    python eval_images.py prepare --per-label 50

Unseen test set:
    python build_image_set.py ammeba  --tag test --seed 2026 --exclude data/images.csv
    python build_image_set.py commons --tag test --seed 2026 --exclude data/images.csv
    python build_image_set.py combine --tag test
    python eval_images.py prepare --manifest data/images_test.csv --out data/images_test_prepared.csv --per-label 50
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List
from urllib.parse import urlparse

import eval_common as ec
import eval_images as ei

AMMEBA_OUT = os.path.join("data", "images_ammeba.csv")
COMMONS_OUT = os.path.join("data", "images_commons.csv")
COMBINED = os.path.join("data", "images.csv")
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "thesis-image-eval/1.0 (honours research; python-requests)"

# Hosts whose copies are often stamped or annotated.
SKIP_HOSTS = (
    "snopes.com", "factcheck.afp.com", "boomlive.in", "altnews.in", "leadstories.com", "vishvasnews.com",
    "politifact.com", "fullfact.org", "factly.in", "newsmeter.in", "newsmobile.in", "africacheck.org",
    "misbar.com", "checkyourfact.com", "factcheck.org", "aap.com.au", "reuters.com", "usatoday.com",
    "archive.ph", "archive.is", "archive.fo", "archive.today", "archive.org", "perma.cc",
)

THEMES = [
    "flood", "bushfire", "wildfire", "protest march", "demonstration", "earthquake damage",
    "hurricane damage", "tornado damage", "storm damage", "refugee camp", "election rally",
    "parliament", "military parade", "snowstorm", "volcanic eruption", "crowd rally",
]
GENERIC = re.compile(r"^(own work|photo|image|picture|untitled|img[_ ]?\d+|dsc[_ ]?\d+)\.?$", re.I)


# AMMeBa


def _paths(tag: str):
    sfx = f"_{tag}" if tag else ""
    return (os.path.join("data", f"images_ammeba{sfx}.csv"), os.path.join("data", f"images_commons{sfx}.csv"),
            os.path.join("data", f"images{sfx}.csv"))


def _used(paths: List[str]) -> dict:
    """Image ids, URLs and Commons pages already used by an earlier list."""
    used = {"ammeba": set(), "urls": set(), "pages": set()}
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"--exclude file not found: {p}")
        for r in ec.read_csv(p):
            if r.get("image_url"):
                used["urls"].add(r["image_url"])
            notes = r.get("notes") or ""
            m = re.search(r"image_id=(\S+)", notes)
            if m:
                used["ammeba"].add(m.group(1))
            used["pages"].update(re.findall(r"https?://commons\.wikimedia\.org/wiki/File:\S+?(?= \(|;|$)", notes))
    return used


def _ok_url(u) -> bool:
    if not isinstance(u, str) or not u.startswith("http"):
        return False
    host = urlparse(u).netloc.lower()
    if any(host == h or host.endswith("." + h) for h in SKIP_HOSTS):
        return False
    if "fbcdn" in host and "oe=" in u:
        return False
    return True


def ammeba(args) -> None:
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Needs pandas: pip install pandas")
    s2_path, im_path = os.path.join(args.archive, "stage_2.csv"), os.path.join(args.archive, "image_metadata.csv")
    for p in (s2_path, im_path):
        if not os.path.exists(p):
            sys.exit(f"{p} not found. Put the AMMeBa Kaggle files in {args.archive}/.")
    s2 = pd.read_csv(s2_path, low_memory=False)
    s2 = s2[s2["disqualified"].astype(str).str.lower() == "false"].copy()

    def category(r):
        cm, cx, sm = str(r.content_manipulation), str(r.context_manipulation), str(r.self_miscontextualizing)
        if cm == "true" and cx != "true":
            return "content_manipulation"
        if cx == "true" and cm == "false" and sm == "true":
            return "miscontextualised"
        return "other"

    s2["cat"] = s2.apply(category, axis=1)
    s2["ai"] = s2["ai_manipulation"].astype(str).str.lower() == "true"
    s2["flagged"] = (s2["image_annotation"].astype(str).str.lower() == "true") | \
                    (s2["image_watermark"].astype(str).str.lower() == "true")
    g = s2.groupby("image_id").agg(n=("cat", "size"), cats=("cat", lambda x: set(x)), ai=("ai", "max"),
                                   flagged=("flagged", "max"), fact_check_url=("fact_check_url", "first"))
    g = g[(g["n"] >= 2) & g["cats"].map(lambda c: len(c) == 1 and "other" not in c) & ~g["flagged"]]
    g["label"] = g["cats"].map(lambda c: next(iter(c)))
    im = pd.read_csv(im_path).set_index("image_id")
    g = g.join(im[["url", "mime", "width", "height"]], how="inner")
    g = g[g["url"].map(_ok_url) & g["mime"].isin(["image/jpeg", "image/png"])]
    if args.exclude:
        used = _used(args.exclude)
        before = len(g)
        g = g[~g.index.astype(str).isin(used["ammeba"]) & ~g["url"].isin(used["urls"])]
        print(f"Excluded {before - len(g)} AMMeBa images already used in {', '.join(args.exclude)}.")
    pfx = f"{args.tag}_" if args.tag else ""

    rnd = random.Random(args.seed)
    n_cand = args.per_label * args.oversample
    rows = []
    # alternate AI-generated and edited images
    cm = g[g["label"] == "content_manipulation"]
    ai_ids, edit_ids = list(cm[cm["ai"]].index), list(cm[~cm["ai"]].index)
    rnd.shuffle(ai_ids)
    rnd.shuffle(edit_ids)
    order = []
    while (ai_ids or edit_ids) and len(order) < n_cand:
        for pool in (ai_ids, edit_ids):
            if pool and len(order) < n_cand:
                order.append(pool.pop())
    for iid in order:
        r = g.loc[iid]
        rows.append(_row(iid, "content_manipulation", r, "ai_generated" if r["ai"] else "edited", pfx))
    mc_ids = list(g[g["label"] == "miscontextualised"].index)
    rnd.shuffle(mc_ids)
    for iid in mc_ids[: max(1, n_cand // 2)]:  # Commons supplies the other half
        rows.append(_row(iid, "miscontextualised", g.loc[iid], "self_captioned", pfx))
    out = _paths(args.tag)[0]
    ec.write_csv(out, rows, ei.FIELDS + ["group"])
    print(f"Wrote {len(rows)} AMMeBa candidates to {out}: {dict(Counter(r['group'] for r in rows))}")
    print(f"(Eligible pool: {dict(g['label'].value_counts())}, of which AI-generated: {int(g[g['label'] == 'content_manipulation']['ai'].sum())})")


def _row(iid, label, r, group, pfx="") -> dict:
    return {"id": f"{pfx}ammeba_{str(iid)[:12]}", "label": label, "caption": "", "image_url": r["url"],
            "source": "AMMeBa", "fact_check_url": r["fact_check_url"], "group": group,
            "notes": f"image_id={iid}"}


# Wikimedia Commons


def _get(params: dict) -> dict:
    import requests

    for attempt in range(4):
        r = requests.get(API, params={**params, "format": "json", "formatversion": 2},
                         headers={"User-Agent": USER_AGENT}, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def _strip(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _search(theme: str, max_pages: int) -> List[dict]:
    params = {"action": "query", "generator": "search", "gsrsearch": f"{theme} filetype:bitmap",
              "gsrnamespace": 6, "gsrlimit": 50, "prop": "imageinfo",
              "iiprop": "url|extmetadata|size|mime", "iiurlwidth": 1200}
    found, cont = [], {}
    for _ in range(max_pages):
        data = _get({**params, **cont})
        found.extend(data.get("query", {}).get("pages", []))
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(0.5)
    return found


def _usable(page: dict, theme: str):
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata") or {}
    if info.get("mime") not in ("image/jpeg", "image/png") or (info.get("width") or 0) < 800:
        return None
    desc = _strip((meta.get("ImageDescription") or {}).get("value", ""))
    date = _strip((meta.get("DateTimeOriginal") or {}).get("value", ""))
    year = re.search(r"\b(19[5-9]\d|20[0-4]\d)\b", date or desc)
    if not year or not (30 <= len(desc) <= 400) or GENERIC.match(desc) or "http" in desc.lower():
        return None
    day = re.match(r"(\d{4}-\d{2}-\d{2})", date or "")
    caption = f"{desc.rstrip('.')}. Photographed {day.group(1) if day else 'in ' + year.group(1)}."
    return {"title": page.get("title", ""), "year": int(year.group(1)), "caption": caption,
            "image_url": info.get("thumburl") or info.get("url"), "page": info.get("descriptionurl", ""),
            "licence": _strip((meta.get("LicenseShortName") or {}).get("value", "")),
            "artist": _strip((meta.get("Artist") or {}).get("value", ""))[:80], "theme": theme}


def commons(args) -> None:
    if args.test:
        data = _get({"action": "query", "generator": "search", "gsrsearch": "flood filetype:bitmap",
                     "gsrnamespace": 6, "gsrlimit": 2, "prop": "imageinfo",
                     "iiprop": "url|extmetadata|size|mime", "iiurlwidth": 1200})
        print(json.dumps(data, indent=1)[:3000])
        return
    rnd = random.Random(args.seed)
    by_theme: Dict[str, List[dict]] = defaultdict(list)
    seen = set()
    used = _used(args.exclude) if args.exclude else {"urls": set(), "pages": set()}
    skipped = 0
    pfx = f"{args.tag}_" if args.tag else ""
    for theme in args.themes:
        pages = _search(theme, args.pages)
        for p in pages:
            item = _usable(p, theme)
            if item and (item["image_url"] in used["urls"] or item["page"] in used["pages"]):
                skipped += 1
                continue
            if item and item["title"] not in seen:
                seen.add(item["title"])
                by_theme[theme].append(item)
        print(f"  {theme:<20} {len(by_theme[theme])} usable photos")
    # skip descriptions shared by several files (batch captions)
    desc_n = Counter(it["caption"].split(". Photographed")[0] for items in by_theme.values() for it in items)
    shared = 0
    for theme in list(by_theme):
        keep = [it for it in by_theme[theme] if desc_n[it["caption"].split(". Photographed")[0]] == 1]
        shared += len(by_theme[theme]) - len(keep)
        by_theme[theme] = keep
    print(f"Dropped {shared} photos whose description is shared with other files (series captions).")
    if args.exclude:
        print(f"Excluded {skipped} Commons photos already used in {', '.join(args.exclude)}.")
    for items in by_theme.values():
        rnd.shuffle(items)

    n_true = args.per_label * args.oversample
    n_swap = max(1, (args.per_label * args.oversample) // 2)
    rows, swaps = [], []
    themes = [t for t in by_theme if by_theme[t]]
    # caption swaps: same theme, different year
    t_i = 0
    while len(swaps) < n_swap and any(len(by_theme[t]) >= 2 for t in themes):
        theme = themes[t_i % len(themes)]
        t_i += 1
        items = by_theme[theme]
        if len(items) < 2:
            continue
        img = items.pop()
        donor_idx = next((k for k, d in enumerate(items) if abs(d["year"] - img["year"]) >= 2), None)
        if donor_idx is None:
            continue
        donor = items.pop(donor_idx)
        swaps.append({"id": f"{pfx}commons_swap_{len(swaps) + 1:03d}", "label": "miscontextualised",
                      "caption": donor["caption"], "image_url": img["image_url"], "source": "Wikimedia Commons",
                      "group": "caption_swap",
                      "notes": f"image {img['page']} ({img['year']}; {img['licence']}; {img['artist']}); "
                               f"caption from {donor['page']} ({donor['year']})"})
    pool = [it for t in themes for it in by_theme[t]]
    rnd.shuffle(pool)
    for k, it in enumerate(pool[:n_true], start=1):
        rows.append({"id": f"{pfx}commons_true_{k:03d}", "label": "authentic_truthful", "caption": it["caption"],
                     "image_url": it["image_url"], "source": "Wikimedia Commons", "group": "original_caption",
                     "notes": f"{it['page']} ({it['licence']}; {it['artist']})"})
    rows += swaps
    out = _paths(args.tag)[1]
    ec.write_csv(out, rows, ei.FIELDS + ["group"])
    print(f"Wrote {len(rows)} Commons candidates to {out}: {dict(Counter(r['group'] for r in rows))}")
    if len(swaps) < n_swap:
        print("Fewer caption swaps than planned; add themes with --themes or raise --pages.")


def combine(args) -> None:
    """Interleave the sources within each category."""
    a_out, c_out, combined = _paths(args.tag)
    parts = [ec.read_csv(p) for p in (a_out, c_out) if os.path.exists(p)]
    if not parts:
        sys.exit("Run `ammeba` and/or `commons` first.")
    by_label: Dict[str, List[List[dict]]] = defaultdict(list)
    for part in parts:
        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in part:
            groups[r["label"]].append(r)
        for lab, rs in groups.items():
            by_label[lab].append(rs)
    rows = []
    for lab in ei.GOLD:
        lists = by_label.get(lab, [])
        while any(lists):
            for lst in lists:
                if lst:
                    rows.append(lst.pop(0))
    ec.write_csv(combined, rows, ei.FIELDS + ["group"])
    print(f"Wrote {len(rows)} candidates to {combined}: {dict(Counter(r['label'] for r in rows))}")
    if args.tag:
        print(f"Next: python eval_images.py prepare --manifest {combined} "
              f"--out data/images_{args.tag}_prepared.csv --per-label 50")
    else:
        print("Next: python eval_images.py prepare --per-label 50")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the stage 2 image list (free).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ammeba", help="Candidates from the AMMeBa Kaggle files.")
    a.add_argument("--archive", default="archive")
    a.add_argument("--per-label", type=int, default=50)
    a.add_argument("--oversample", type=int, default=3, help="Candidates per image needed (dead links).")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--tag", default="", help="Name for a second set, e.g. test (separate files and ids).")
    a.add_argument("--exclude", nargs="*", default=[], help="Image lists whose images must not be reused.")
    c = sub.add_parser("commons", help="Candidates from Wikimedia Commons.")
    c.add_argument("--per-label", type=int, default=50)
    c.add_argument("--oversample", type=int, default=2)
    c.add_argument("--pages", type=int, default=3, help="Result pages (50 files each) per theme.")
    c.add_argument("--themes", nargs="*", default=THEMES)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--test", action="store_true", help="Print one raw API reply and stop.")
    c.add_argument("--tag", default="", help="Name for a second set, e.g. test (separate files and ids).")
    c.add_argument("--exclude", nargs="*", default=[], help="Image lists whose images and captions must not be reused.")
    m = sub.add_parser("combine", help="Merge both into data/images.csv.")
    m.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    {"ammeba": ammeba, "commons": commons, "combine": combine}[args.cmd](args)


if __name__ == "__main__":
    main()
