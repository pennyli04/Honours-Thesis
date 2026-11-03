# Stage 2 image evaluation

Generated 12 October 2026, 14:33. Categories follow Dufour et al. (2024).

| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Fact-check page returned | Cost (USD) | Reverse searches |
|---|---|---|---|---|---|---|---|
| full_test | 150 | 78.0% (70.7 to 84.7) | 59.1% (53.8 to 80.1) | 98.0% | 18.0% | 4.46 | 181 |
| full_test_nosearch | 150 | 77.3% (70.0 to 84.0) | 59.7% (54.5 to 64.3) | 92.7% | 0.7% | 5.24 | 0 |
| classifier_only_test | 150 | 33.3% (26.0 to 40.7) | 17.8% (14.0 to 21.6) | 100.0% | 0.0% | 0.00 | 0 |

## Accuracy by gold category

| Condition | Content manipulation | Miscontextualised | Authentic and truthful |
|---|---|---|---|
| full_test | 72.0% (n=50) | 74.0% (n=50) | 88.0% (n=50) |
| full_test_nosearch | 60.0% (n=50) | 76.0% (n=50) | 96.0% (n=50) |
| classifier_only_test | 98.0% (n=50) | 0.0% (n=50) | 2.0% (n=50) |

## Accuracy by source and type of item

| Condition | Gold category | Source / type | n | Accuracy |
|---|---|---|---|---|
| full_test | authentic_truthful | Wikimedia Commons / original_caption | 50 | 88.0% |
| full_test | content_manipulation | AMMeBa / ai_generated | 10 | 100.0% |
| full_test | content_manipulation | AMMeBa / edited | 40 | 65.0% |
| full_test | miscontextualised | AMMeBa / self_captioned | 23 | 52.2% |
| full_test | miscontextualised | Wikimedia Commons / caption_swap | 27 | 92.6% |
| full_test_nosearch | authentic_truthful | Wikimedia Commons / original_caption | 50 | 96.0% |
| full_test_nosearch | content_manipulation | AMMeBa / ai_generated | 10 | 80.0% |
| full_test_nosearch | content_manipulation | AMMeBa / edited | 40 | 55.0% |
| full_test_nosearch | miscontextualised | AMMeBa / self_captioned | 23 | 47.8% |
| full_test_nosearch | miscontextualised | Wikimedia Commons / caption_swap | 27 | 100.0% |
| classifier_only_test | authentic_truthful | Wikimedia Commons / original_caption | 50 | 2.0% |
| classifier_only_test | content_manipulation | AMMeBa / ai_generated | 10 | 100.0% |
| classifier_only_test | content_manipulation | AMMeBa / edited | 40 | 97.5% |
| classifier_only_test | miscontextualised | AMMeBa / self_captioned | 23 | 0.0% |
| classifier_only_test | miscontextualised | Wikimedia Commons / caption_swap | 27 | 0.0% |

## Stated confidence (model conditions)

| Condition | High | Medium | Low |
|---|---|---|---|
| full_test | 94.7% (n=57) | 69.5% (n=82) | 54.5% (n=11) |
| full_test_nosearch | 100.0% (n=18) | 84.5% (n=110) | 22.7% (n=22) |

## Paired comparison (same images, exact McNemar test)

| A | B | Images | Accuracy A | Accuracy B | Only A right | Only B right | p |
|---|---|---|---|---|---|---|---|
| full_test | full_test_nosearch | 150 | 78.0% | 77.3% | 11 | 10 | 1.0000 |
| full_test | classifier_only_test | 150 | 78.0% | 33.3% | 81 | 14 | 0.0000 |
| full_test_nosearch | classifier_only_test | 150 | 77.3% | 33.3% | 86 | 20 | 0.0000 |

The classifier-only condition can only answer content manipulation or authentic, so it cannot be right on miscontextualised images by design.
