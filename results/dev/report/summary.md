# AVeriTeC evaluation (check 1: claim verification)

Generated 28 September 2026, 15:19 with model `claude-sonnet-5`. Verdict label only; not comparable with the official AVeriTeC evidence-weighted score.

Prompt versions: search_blocked: v2-misleading-attribution

## Headline results

| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Selective accuracy | Own fact-check retrieved |
|---|---|---|---|---|---|---|
| search_blocked | 158 | 65.8% (58.2 to 72.8) | 47.1% (39.9 to 54.6) | 98.1% | 66.5% | 1.3% |

Coverage is the share of claims given a definite verdict (true, false or misleading); selective accuracy is accuracy on those claims alone.

## Per-class results

**search_blocked**

| Predicted class | Precision | Recall | F1 | Gold support |
|---|---|---|---|---|
| likely_true | 76.9% | 75.0% | 75.9% | 40 |
| likely_false | 90.7% | 73.1% | 81.0% | 93 |
| misleading | 12.2% | 41.7% | 18.9% | 12 |
| unverifiable | 33.3% | 7.7% | 12.5% | 13 |

## Confusion matrices (rows: AVeriTeC gold label, columns: prediction)

**search_blocked**

| Gold | likely_true | likely_false | misleading | unverifiable |
|---|---|---|---|---|
| Supported | 30 | 1 | 9 | 0 |
| Refuted | 5 | 68 | 18 | 2 |
| Not Enough Evidence | 2 | 1 | 9 | 1 |
| Conflicting Evidence/Cherrypicking | 2 | 5 | 5 | 0 |

## Accuracy by stated confidence

| Condition | High | Medium | Low |
|---|---|---|---|
| search_blocked | 74.4% (n=117) | 43.6% (n=39) | 0.0% (n=2) |

## Leakage

| Condition | Own fact-check retrieved | Any fact-check page retrieved | Justification cites a fact-checker | Accuracy with a fact-check page | Accuracy without (n) |
|---|---|---|---|---|---|
| search_blocked | 1.3% | 55.7% | 18.4% | 69.3% | 61.4% (70) |

A fact-check page is any retrieved URL from a blocked site or whose address marks it as a fact-check (for example cnn.com/.../biden-fracking-fact-check). Citing a fact-checker in the justification without retrieving its page points to knowledge from training or an indirect source, which blocking cannot prevent.

## Cost and time per claim

| Condition | Input tokens | Output tokens | Web searches | Seconds | Est. cost (USD) | Follow-up needed |
|---|---|---|---|---|---|---|
| search_blocked | 48511 | 1679 | 2.6 | 19.8 | set --price-in/--price-out | 8.2% |

Claims excluded because of API or runtime errors: {'search_blocked': 42}. Re-run with --retry-errors.
