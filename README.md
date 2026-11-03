# Staged agentic misinformation detection

Code and results for my honours thesis, *A Staged Agentic AI Workflow for Detecting Misinformation and Supporting News Credibility Judgement* (University of Technology Sydney, 2026).

The checker uses one language model (Claude Sonnet 5) in three stages:

1. **Text claims:** web search with known fact checking sites blocked.
2. **Images:** reverse image search (SerpApi Google Lens), a local deepfake classifier and the model's own reading of the picture.
3. **Provenance:** C2PA Content Credentials read with c2pa-python.

Each stage was evaluated against a baseline without the added capability. The thesis reports the results.

## Folder layout

| Path | What it holds |
|---|---|
| `factcheck.py` | The checker itself (all three stages) |
| `eval_averitec.py` | Stage 1 evaluation on AVeriTeC and on the post-cutoff claims |
| `eval_images.py` | Stage 2 evaluation |
| `eval_c2pa.py` | Stage 3 evaluation on a controlled C2PA set |
| `eval_common.py` | Shared helpers for stages 2 and 3 |
| `build_image_set.py` | Builds the image lists from AMMeBa and Wikimedia Commons |
| `build_recent_claims.py` | Builds the post-cutoff claim set from the Google Fact Check Tools API |
| `analyse_averitec.py` | Extra analysis and figures from saved stage 1 results |
| `plot_figure10.py` | Figure 10 of the thesis from the saved results |
| `test_factcheck.py` | Offline tests (no API calls) |
| `data/` | Image lists and the post-cutoff claim set (images themselves are not included) |
| `results/` | Every run reported in the thesis, one JSON line per item, plus score reports |
| `c2pa_test_certs/` | The public c2pa-python test certificate used to sign the stage 3 set (see its README) |

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
cp env.example .env
```

Fill in `.env`:

- `ANTHROPIC_API_KEY` for all paid runs
- `SERPAPI_API_KEY` for reverse image search (stage 2)
- `GOOGLE_FACTCHECK_API_KEY` only for rebuilding the post-cutoff claim set

Never commit `.env`. It is listed in `.gitignore`.

## Checking the reported numbers without spending anything

All results are in `results/`, so the tables and figures can be rebuilt for free:

```
python eval_averitec.py score
python eval_images.py score
python eval_c2pa.py score
python plot_figure10.py
python -m pytest -q test_factcheck.py
```

## Rerunning the experiments

Every paid run prints a cost estimate and asks before sending anything. Add `--dry-run` to see only the estimate.

Stage 1 (AVeriTeC held out sample):

```
python eval_averitec.py download
python eval_averitec.py run --condition search_blocked --sample 150 --batch
python eval_averitec.py run --condition no_search --sample 150 --batch
```

Stage 1 (claims made after the training data):

```
python build_recent_claims.py fetch
python eval_averitec.py run --data data/recent_claims_sample.json --condition search_blocked --batch --tag recent
python eval_averitec.py run --data data/recent_claims_sample.json --condition no_search --batch --tag recent
```

Stage 2 (the images are downloaded again from the links in the lists):

```
python eval_images.py prepare --manifest data/images_test.csv --out data/images_test_prepared.csv --per-label 50
python eval_images.py run --condition classifier_only --manifest data/images_test_prepared.csv --tag test
python eval_images.py run --condition full --manifest data/images_test_prepared.csv --tag test
```

To rebuild the image lists from scratch, put the AMMeBa Kaggle files (`stage_2.csv`, `image_metadata.csv`) in `archive/` and see the commands at the top of `build_image_set.py`.

Stage 3:

```
python eval_c2pa.py make-set
python eval_c2pa.py check
python eval_c2pa.py run --condition rule
python eval_c2pa.py run --condition model
```

## Data sources

- AVeriTeC (Schlichtkrull et al., 2023), downloaded by `eval_averitec.py download` and not redistributed here
- AMMeBa (Dufour et al., 2024), Kaggle release
- Wikimedia Commons, openly licensed photographs
- Google Fact Check Tools API (ClaimReview data from the fact checkers listed in `build_recent_claims.py`)

Images from AMMeBa are misinformation as it circulated online and belong to their original owners, so only their links are included.
