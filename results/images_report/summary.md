# Stage 2 image evaluation

Generated 06 October 2026, 16:41. Categories follow Dufour et al. (2024).

| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Fact-check page returned | Cost (USD) | Reverse searches |
|---|---|---|---|---|---|---|---|
| full | 150 | 72.0% (64.0 to 79.3) | 57.2% (51.9 to 61.8) | 86.0% | 13.3% | 3.74 | 175 |
| no_image | 150 | 51.3% (43.3 to 59.3) | 42.7% (36.2 to 48.1) | 73.3% | 13.3% | 2.98 | 167 |
| classifier_only | 150 | 33.3% (26.0 to 40.7) | 17.9% (14.4 to 21.8) | 100.0% | 0.0% | 0.00 | 0 |

## Accuracy by gold category

| Condition | Content manipulation | Miscontextualised | Authentic and truthful |
|---|---|---|---|
| full | 82.0% (n=50) | 88.0% (n=50) | 46.0% (n=50) |
| no_image | 82.0% (n=50) | 30.0% (n=50) | 42.0% (n=50) |
| classifier_only | 98.0% (n=50) | 0.0% (n=50) | 2.0% (n=50) |

## Accuracy by source and type of item

| Condition | Gold category | Source / type | n | Accuracy |
|---|---|---|---|---|
| full | authentic_truthful | Wikimedia Commons / original_caption | 50 | 46.0% |
| full | content_manipulation | AMMeBa / ai_generated | 25 | 96.0% |
| full | content_manipulation | AMMeBa / edited | 25 | 68.0% |
| full | miscontextualised | AMMeBa / self_captioned | 24 | 83.3% |
| full | miscontextualised | Wikimedia Commons / caption_swap | 26 | 92.3% |
| no_image | authentic_truthful | Wikimedia Commons / original_caption | 50 | 42.0% |
| no_image | content_manipulation | AMMeBa / ai_generated | 25 | 96.0% |
| no_image | content_manipulation | AMMeBa / edited | 25 | 68.0% |
| no_image | miscontextualised | AMMeBa / self_captioned | 24 | 37.5% |
| no_image | miscontextualised | Wikimedia Commons / caption_swap | 26 | 23.1% |
| classifier_only | authentic_truthful | Wikimedia Commons / original_caption | 50 | 2.0% |
| classifier_only | content_manipulation | AMMeBa / ai_generated | 25 | 96.0% |
| classifier_only | content_manipulation | AMMeBa / edited | 25 | 100.0% |
| classifier_only | miscontextualised | AMMeBa / self_captioned | 24 | 0.0% |
| classifier_only | miscontextualised | Wikimedia Commons / caption_swap | 26 | 0.0% |

## Stated confidence (model conditions)

| Condition | High | Medium | Low |
|---|---|---|---|
| full | 98.0% (n=49) | 76.3% (n=76) | 8.0% (n=25) |
| no_image | 100.0% (n=14) | 68.5% (n=89) | 4.3% (n=47) |

## Paired comparison (same images, exact McNemar test)

| A | B | Images | Accuracy A | Accuracy B | Only A right | Only B right | p |
|---|---|---|---|---|---|---|---|
| full | no_image | 150 | 72.0% | 51.3% | 35 | 4 | 0.0000 |
| full | classifier_only | 150 | 72.0% | 33.3% | 68 | 10 | 0.0000 |
| no_image | classifier_only | 150 | 51.3% | 33.3% | 37 | 10 | 0.0001 |

The classifier-only condition can only answer content manipulation or authentic, so it cannot be right on miscontextualised images by design.
