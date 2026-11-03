# AVeriTeC evaluation (check 1: claim verification)

Generated 12 October 2026, 16:48 with model `claude-sonnet-5`. Verdict label only; not comparable with the official AVeriTeC evidence-weighted score.

Prompt versions: search_blocked_recent: v3-central-assertion; no_search_recent: v3-central-assertion

## Headline results

| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Selective accuracy | Own fact-check retrieved |
|---|---|---|---|---|---|---|
| search_blocked_recent | 150 | 60.0% (52.0 to 67.3) | 48.1% (40.3 to 55.9) | 96.7% | 60.7% | 0.0% |
| no_search_recent | 150 | 28.7% (22.0 to 36.0) | 25.4% (19.2 to 31.5) | 38.7% | 58.6% | 0.0% |

Coverage is the share of claims given a definite verdict (true, false or misleading); selective accuracy is accuracy on those claims alone.

## Per-class results

**search_blocked_recent**

| Predicted class | Precision | Recall | F1 | Gold support |
|---|---|---|---|---|
| likely_true | 66.7% | 86.5% | 75.3% | 37 |
| likely_false | 59.3% | 94.1% | 72.7% | 51 |
| misleading | 50.0% | 17.8% | 26.2% | 45 |
| unverifiable | 40.0% | 11.8% | 18.2% | 17 |

**no_search_recent**

| Predicted class | Precision | Recall | F1 | Gold support |
|---|---|---|---|---|
| likely_true | 70.0% | 18.9% | 29.8% | 37 |
| likely_false | 57.4% | 52.9% | 55.1% | 51 |
| misleading | 0.0% | 0.0% | 0.0% | 45 |
| unverifiable | 9.8% | 52.9% | 16.5% | 17 |

## Confusion matrices (rows: AVeriTeC gold label, columns: prediction)

**search_blocked_recent**

| Gold | likely_true | likely_false | misleading | unverifiable |
|---|---|---|---|---|
| Supported | 32 | 1 | 4 | 0 |
| Refuted | 0 | 48 | 3 | 0 |
| Not Enough Evidence | 0 | 14 | 1 | 2 |
| Conflicting Evidence/Cherrypicking | 16 | 18 | 8 | 3 |

**no_search_recent**

| Gold | likely_true | likely_false | misleading | unverifiable |
|---|---|---|---|---|
| Supported | 7 | 3 | 1 | 26 |
| Refuted | 2 | 27 | 0 | 22 |
| Not Enough Evidence | 0 | 8 | 0 | 9 |
| Conflicting Evidence/Cherrypicking | 1 | 9 | 0 | 35 |

## Accuracy by stated confidence

| Condition | High | Medium | Low |
|---|---|---|---|
| search_blocked_recent | 66.7% (n=126) | 27.3% (n=22) | 0.0% (n=2) |
| no_search_recent | 34.5% (n=29) | 56.4% (n=39) | 13.4% (n=82) |

## Leakage

| Condition | Own fact-check retrieved | Any fact-check page retrieved | Justification cites a fact-checker | Accuracy with a fact-check page | Accuracy without (n) |
|---|---|---|---|---|---|
| search_blocked_recent | 0.0% | 48.0% | 24.7% | 66.7% | 53.8% (78) |
| no_search_recent | 0.0% | 0.0% | 0.7% | n/a | 28.7% (150) |

A fact-check page is any retrieved URL from a blocked site or whose address marks it as a fact-check (for example cnn.com/.../biden-fracking-fact-check). Citing a fact-checker in the justification without retrieving its page points to knowledge from training or an indirect source, which blocking cannot prevent.

## Paired comparison (same claims, exact McNemar test)

| A | B | Claims | Accuracy A | Accuracy B | Only A right | Only B right | p |
|---|---|---|---|---|---|---|---|
| search_blocked_recent | no_search_recent | 150 | 60.0% | 28.7% | 55 | 8 | 0.0000 |

## Cost and time per claim

| Condition | Input tokens | Output tokens | Web searches (cap) | Seconds | Batch API | Cost per claim (USD) | Total spent (USD) | Follow-up needed |
|---|---|---|---|---|---|---|---|---|
| search_blocked_recent | 34481 | 1480 | 2.2 (3) | 0.0 | 100.0% | 0.0641 | 9.62 | 0.0% |
| no_search_recent | 1059 | 670 | 0.0 (0) | 0.0 | 100.0% | 0.0044 | 0.66 | 0.0% |

Costs are estimates from token and search counts at the list prices in eval_averitec.py (checked 28 September 2026); the Anthropic Console shows the exact amount billed.
