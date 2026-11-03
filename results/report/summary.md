# AVeriTeC evaluation (check 1: claim verification)

Generated 06 October 2026, 17:46 with model `claude-sonnet-5`. Verdict label only; not comparable with the official AVeriTeC evidence-weighted score.

Prompt versions: no_search_heldout: v3-central-assertion; search_blocked_heldout: v3-central-assertion

## Headline results

| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Selective accuracy | Own fact-check retrieved |
|---|---|---|---|---|---|---|
| no_search_heldout | 150 | 72.7% (65.3 to 79.3) | 55.1% (45.0 to 64.1) | 90.0% | 77.8% | 0.0% |
| search_blocked_heldout | 150 | 74.0% (66.7 to 80.7) | 50.4% (41.3 to 59.5) | 97.3% | 75.3% | 2.7% |

Coverage is the share of claims given a definite verdict (true, false or misleading); selective accuracy is accuracy on those claims alone.

## Per-class results

**no_search_heldout**

| Predicted class | Precision | Recall | F1 | Gold support |
|---|---|---|---|---|
| likely_true | 74.3% | 72.2% | 73.2% | 36 |
| likely_false | 87.2% | 81.5% | 84.3% | 92 |
| misleading | 28.6% | 36.4% | 32.0% | 11 |
| unverifiable | 26.7% | 36.4% | 30.8% | 11 |

**search_blocked_heldout**

| Predicted class | Precision | Recall | F1 | Gold support |
|---|---|---|---|---|
| likely_true | 67.5% | 75.0% | 71.1% | 36 |
| likely_false | 86.8% | 85.9% | 86.3% | 92 |
| misleading | 26.7% | 36.4% | 30.8% | 11 |
| unverifiable | 25.0% | 9.1% | 13.3% | 11 |

## Confusion matrices (rows: AVeriTeC gold label, columns: prediction)

**no_search_heldout**

| Gold | likely_true | likely_false | misleading | unverifiable |
|---|---|---|---|---|
| Supported | 26 | 5 | 3 | 2 |
| Refuted | 2 | 75 | 6 | 9 |
| Not Enough Evidence | 3 | 3 | 1 | 4 |
| Conflicting Evidence/Cherrypicking | 4 | 3 | 4 | 0 |

**search_blocked_heldout**

| Gold | likely_true | likely_false | misleading | unverifiable |
|---|---|---|---|---|
| Supported | 27 | 5 | 4 | 0 |
| Refuted | 6 | 79 | 4 | 3 |
| Not Enough Evidence | 4 | 3 | 3 | 1 |
| Conflicting Evidence/Cherrypicking | 3 | 4 | 4 | 0 |

## Accuracy by stated confidence

| Condition | High | Medium | Low |
|---|---|---|---|
| no_search_heldout | 89.7% (n=58) | 72.7% (n=66) | 34.6% (n=26) |
| search_blocked_heldout | 78.8% (n=113) | 59.5% (n=37) | n/a (n=0) |

## Leakage

| Condition | Own fact-check retrieved | Any fact-check page retrieved | Justification cites a fact-checker | Accuracy with a fact-check page | Accuracy without (n) |
|---|---|---|---|---|---|
| no_search_heldout | 0.0% | 0.0% | 12.7% | n/a | 72.7% (150) |
| search_blocked_heldout | 2.7% | 51.3% | 22.7% | 77.9% | 69.9% (73) |

A fact-check page is any retrieved URL from a blocked site or whose address marks it as a fact-check (for example cnn.com/.../biden-fracking-fact-check). Citing a fact-checker in the justification without retrieving its page points to knowledge from training or an indirect source, which blocking cannot prevent.

## Paired comparison (same claims, exact McNemar test)

| A | B | Claims | Accuracy A | Accuracy B | Only A right | Only B right | p |
|---|---|---|---|---|---|---|---|
| no_search_heldout | search_blocked_heldout | 150 | 72.7% | 74.0% | 11 | 13 | 0.8388 |

## Cost and time per claim

| Condition | Input tokens | Output tokens | Web searches (cap) | Seconds | Batch API | Cost per claim (USD) | Total spent (USD) | Follow-up needed |
|---|---|---|---|---|---|---|---|---|
| no_search_heldout | 1089 | 721 | 0.0 (0) | 0.0 | 100.0% | 0.0047 | 0.71 | 0.7% |
| search_blocked_heldout | 31329 | 1563 | 2.2 (3) | 0.0 | 100.0% | 0.0612 | 9.18 | 0.0% |

Costs are estimates from token and search counts at the list prices in eval_averitec.py (checked 28 September 2026); the Anthropic Console shows the exact amount billed.
