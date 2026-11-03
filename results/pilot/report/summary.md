# AVeriTeC evaluation (check 1: claim verification)

Generated 28 September 2026, 12:56 with model `claude-sonnet-5`. Verdict label only; not comparable with the official AVeriTeC evidence-weighted score.

## Headline results

| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Selective accuracy | Own fact-check retrieved |
|---|---|---|---|---|---|---|
| search_blocked | 20 | 70.0% (50.0 to 90.0) | 49.5% (34.6 to 74.4) | 100.0% | 70.0% | 0.0% |

Coverage is the share of claims given a definite true or false verdict; selective accuracy is accuracy on those claims alone.

## Per-class results

**search_blocked**

| Predicted class | Precision | Recall | F1 | Gold support |
|---|---|---|---|---|
| likely_true | 50.0% | 100.0% | 66.7% | 5 |
| likely_false | 90.0% | 75.0% | 81.8% | 12 |
| unverifiable | 0.0% | 0.0% | 0.0% | 3 |

## Confusion matrices (rows: AVeriTeC gold label, columns: prediction)

**search_blocked**

| Gold | likely_true | likely_false | unverifiable |
|---|---|---|---|
| Supported | 5 | 0 | 0 |
| Refuted | 3 | 9 | 0 |
| Not Enough Evidence | 0 | 1 | 0 |
| Conflicting Evidence/Cherrypicking | 2 | 0 | 0 |

## Accuracy by stated confidence

| Condition | High | Medium | Low |
|---|---|---|---|
| search_blocked | 68.8% (n=16) | 75.0% (n=4) | n/a (n=0) |

## Leakage

| Condition | Own fact-check retrieved | Any fact-check page retrieved | Justification cites a fact-checker | Accuracy with a fact-check page | Accuracy without (n) |
|---|---|---|---|---|---|
| search_blocked | 0.0% | 55.0% | 25.0% | 63.6% | 77.8% (9) |

A fact-check page is any retrieved URL from a blocked site or whose address marks it as a fact-check (for example cnn.com/.../biden-fracking-fact-check). Citing a fact-checker in the justification without retrieving its page points to knowledge from training or an indirect source, which blocking cannot prevent.

## Cost and time per claim

| Condition | Input tokens | Output tokens | Web searches | Seconds | Est. cost (USD) | Follow-up needed |
|---|---|---|---|---|---|---|
| search_blocked | 31485 | 1416 | 2.2 | 16.4 | set --price-in/--price-out | 0.0% |
