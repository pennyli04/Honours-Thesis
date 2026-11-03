# Stage 3 C2PA evaluation

Generated 06 October 2026, 16:11.

Always answering the most common correct category (unverifiable) would score 60.0%; any useful result must beat that.

| Condition | n | Accuracy (95% CI) | Macro F1 (95% CI) | Cost (USD) |
|---|---|---|---|---|
| rule | 60 | 80.0% (70.0 to 90.0) | 60.0% (56.0 to 63.3) | 0.00 |
| model | 60 | 96.7% (91.7 to 100.0) | 65.7% (64.2 to 100.0) | 0.65 |

## Accuracy by type of file

| File condition | Correct answer | rule | model |
|---|---|---|---|
| signed_capture | unverifiable | 0.0% (n=12) | 83.3% (n=12) |
| ai_generated | content_manipulation | 100.0% (n=12) | 100.0% (n=12) |
| tampered | content_manipulation | 100.0% (n=12) | 100.0% (n=12) |
| stripped | unverifiable | 100.0% (n=12) | 100.0% (n=12) |
| unsigned | unverifiable | 100.0% (n=12) | 100.0% (n=12) |

## Paired comparison (same files, exact McNemar test)

| A | B | Files | Accuracy A | Accuracy B | Only A right | Only B right | p |
|---|---|---|---|---|---|---|---|
| rule | model | 60 | 80.0% | 96.7% | 0 | 10 | 0.0020 |
