"""Draws Figure 10 (stage 1 accuracy with and without search) from the saved
results. No API calls.

    python plot_figure10.py
    python plot_figure10.py --out figures/figure10.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_averitec as ev

SETS = [
    ("AVeriTeC claims from 2020",
     "results/averitec_search_blocked_heldout.jsonl", "results/averitec_no_search_heldout.jsonl"),
    ("Claims made after the training data",
     "results/averitec_search_blocked_recent.jsonl", "results/averitec_no_search_recent.jsonl"),
]

BG, DARK, LIGHT, AXIS, TEXT = "#FCFCFB", "#2768BF", "#A1C4F2", "#BDBDBD", "#555555"


def load(path):
    """Results keyed by claim id, read the same way as eval_averitec.py score."""
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            prev = by_id.get(r["id"])
            if prev is None or prev.get("error") or not r.get("error"):
                by_id[r["id"]] = r
    return {r["id"]: ev.add_derived_flags(r)
            for r in sorted(by_id.values(), key=lambda r: r["id"]) if not r.get("error")}


def summarise(search, closed):
    """Accuracy and 95% CI for each condition, and the paired McNemar p value."""
    s, c = list(search.values()), list(closed.values())
    common = search.keys() & closed.keys()
    only_s = sum(search[k]["correct"] and not closed[k]["correct"] for k in common)
    only_c = sum(closed[k]["correct"] and not search[k]["correct"] for k in common)
    return {
        "n": len(common),
        "search": (ev._accuracy(s) * 100, [v * 100 for v in ev._bootstrap_ci(s, ev._accuracy)]),
        "closed": (ev._accuracy(c) * 100, [v * 100 for v in ev._bootstrap_ci(c, ev._accuracy)]),
        "p": ev._mcnemar_exact(only_s, only_c),
        "discordant": (only_s, only_c),
    }


def p_label(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figure10.png")
    args = ap.parse_args()

    stats = [(name, summarise(load(a), load(b))) for name, a, b in SETS]
    for name, st in stats:
        print(f"{name}: search {st['search'][0]:.1f}% {st['search'][1]}, closed book {st['closed'][0]:.1f}% "
              f"{st['closed'][1]}, discordant {st['discordant']}, {p_label(st['p'])}")

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    width, xs = 0.34, [0, 1.15]
    for i, (name, st) in enumerate(stats):
        x = xs[i]
        tops = []
        for off, key, colour, label in ((-width / 2, "search", DARK, "Search, fact checkers blocked"),
                                        (width / 2, "closed", LIGHT, "Closed book (no search)")):
            val, (lo, hi) = st[key]
            ax.bar(x + off, val, width * 0.92, color=colour, label=label if i == 0 else None, zorder=2)
            ax.errorbar(x + off, val, yerr=[[val - lo], [hi - val]], fmt="none", ecolor="#333333",
                        elinewidth=1.1, capsize=4, zorder=3)
            ax.text(x + off, hi + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10.5, color="#111111")
            tops.append(hi)
        top = max(tops) + 10
        ax.plot([x - width / 2, x - width / 2, x + width / 2, x + width / 2], [top - 2, top, top, top - 2],
                color="#777777", lw=1)
        ax.text(x, top + 1.2, p_label(st["p"]), ha="center", va="bottom", fontsize=10.5, color=TEXT)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{name}\n(n = {st['n']})" for name, st in stats], fontsize=10.5, color=TEXT)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=10.5, colors=TEXT, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Accuracy (%)", loc="left", fontsize=13, color="#111111", pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=2)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=BG)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
