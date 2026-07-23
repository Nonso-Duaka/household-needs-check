# Household Needs Screening — teaching app

A small web app that takes a short household form and returns three
**explainable** outputs: an estimated food-insecurity risk, a transparent
hardship score, and a suggested policy-priority segment.

Built on a **fully synthetic** teaching dataset (100,000 households). It is
oversampled for hardship and **not calibrated to real prevalence** — it must
never be used to make claims about real people, places, or programs, and the
app never presents output as an eligibility decision.

> **Status:** Phase 1 (training + artifacts) is complete. The FastAPI web app
> is Phase 2 and lands after review.

## The point of this project

The honest model is the weaker-looking one. Two fields in the data
(`food_budget_monthly_usd`, `mental_health_stress_score`) are generated
*downstream of the same hardship process that produces the label*, so a model
using them is partly reading its own answer key — and a household can't answer
them on a form anyway. Dropping them takes test ROC AUC from ~0.89 down to
**~0.75**. We ship the ~0.75 model on purpose. That gap is the lesson, and it
gets its own page in the app.

| Feature set | ROC AUC | Precision | Recall |
|---|---|---|---|
| Full 42 predictors | 0.893 | 0.653 | 0.789 |
| minus `food_budget_monthly_usd` | 0.830 | 0.558 | 0.733 |
| **Realistic answerable fields (shipped)** | **0.748** | **0.489** | **0.650** |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at `dataset/synthetic_food_housing_insecurity_100k.csv`.

## Retrain

```bash
python train.py            # trains Model A (food) + Model B (segment)
python train.py --compare  # also fits RF / SVC / HistGBM -> models/comparison.csv
```

Writes versioned artifacts to `models/`:

| File | What it is |
|---|---|
| `food_model.joblib` | Model A — logistic regression, food-insecurity risk |
| `segment_model.joblib` | Model B — multinomial logistic, policy segment |
| `feature_schema.json` | the form↔model contract; **the form renders from this** |
| `metrics.json` | final test metrics, capacity table, recovered-rule report |
| `weights.yaml` | **editable** hardship weights (must sum to 1.0) |
| `comparison.csv` | model bake-off (only with `--compare`) |

The split is the preassigned `train_test_split` column (70,046 / 14,989 /
14,965). We fit on train, inspect thresholds on validation, and report final
numbers on test once.

## What the three outputs are

- **Food-insecurity risk** — an *estimated probability* from Model A, shown with
  a labeled risk band and the decision threshold stated inline. Not a decision.
- **Hardship score (0–100)** — not a model. A transparent weighted sum of
  interpretable components (`hardship.py`, weights in `models/weights.yaml`).
  Each component normalizes to 0–1; the score is their weighted total.
- **Policy-priority segment** — a suggested service segment from Model B, with a
  confidence and a recommended *starting point for review*.

## Derived fields and inferred rules

The form asks 14 questions; three model inputs are computed server-side and
never asked (`rent_or_mortgage_burden_pct`, `income_to_poverty_ratio`,
`food_desert_flag`). Two of those come from a generator whose code we don't
have, so their rules were recovered empirically. What was inferred, and how
confident, is written up in [`notes/derived.md`](notes/derived.md).

## Fairness

`race_ethnicity`, `immigrant_household`, `primary_language`, and
`disability_present` are **deliberately excluded** from the form and from every
prediction. They are audit dimensions only. The training script asserts none of
them (nor any leakage column) is ever used as a predictor.

## Files

```
train.py       Phase 1 training + artifact emission
hardship.py    transparent hardship score + shared feature derivation
models/        artifacts (weights.yaml is tracked; the rest regenerate)
notes/         derived.md — recovered generator rules
```
