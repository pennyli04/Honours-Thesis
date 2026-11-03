# Stage 2 image evaluation

Generated 12 October 2026, 11:47. Categories follow Dufour et al. (2024).

| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Coverage | Fact-check page returned | Cost (USD) | Reverse searches |
|---|---|---|---|---|---|---|---|
| full | 150 | 72.0% (64.0 to 79.3) | 57.2% (51.9 to 61.8) | 86.0% | 13.3% | 3.74 | 175 |
| full_v3dev | 149 | 77.2% (70.5 to 83.9) | 59.5% (54.6 to 64.2) | 94.0% | 9.4% | 4.25 | 120 |

## Accuracy by gold category

| Condition | Content manipulation | Miscontextualised | Authentic and truthful |
|---|---|---|---|
| full | 82.0% (n=50) | 88.0% (n=50) | 46.0% (n=50) |
| full_v3dev | 72.0% (n=50) | 64.0% (n=50) | 95.9% (n=49) |

## Accuracy by source and type of item

| Condition | Gold category | Source / type | n | Accuracy |
|---|---|---|---|---|
| full | authentic_truthful | Wikimedia Commons / original_caption | 50 | 46.0% |
| full | content_manipulation | AMMeBa / ai_generated | 25 | 96.0% |
| full | content_manipulation | AMMeBa / edited | 25 | 68.0% |
| full | miscontextualised | AMMeBa / self_captioned | 24 | 83.3% |
| full | miscontextualised | Wikimedia Commons / caption_swap | 26 | 92.3% |
| full_v3dev | authentic_truthful | Wikimedia Commons / original_caption | 49 | 95.9% |
| full_v3dev | content_manipulation | AMMeBa / ai_generated | 25 | 84.0% |
| full_v3dev | content_manipulation | AMMeBa / edited | 25 | 60.0% |
| full_v3dev | miscontextualised | AMMeBa / self_captioned | 24 | 54.2% |
| full_v3dev | miscontextualised | Wikimedia Commons / caption_swap | 26 | 73.1% |

## Stated confidence (model conditions)

| Condition | High | Medium | Low |
|---|---|---|---|
| full | 98.0% (n=49) | 76.3% (n=76) | 8.0% (n=25) |
| full_v3dev | 86.8% (n=38) | 87.1% (n=93) | 5.6% (n=18) |

## Paired comparison (same images, exact McNemar test)

| A | B | Images | Accuracy A | Accuracy B | Only A right | Only B right | p |
|---|---|---|---|---|---|---|---|
| full | full_v3dev | 149 | 72.5% | 77.2% | 20 | 27 | 0.3817 |

The classifier-only condition can only answer content manipulation or authentic, so it cannot be right on miscontextualised images by design.
