# Household Needs Screening — teaching app

A small web app that takes a short household form and returns three
**explainable** outputs: an estimated food-insecurity risk, a transparent
hardship score, and a suggested policy-priority segment.

Built on a **fully synthetic** teaching dataset (100,000 households). It is
oversampled for hardship and **not calibrated to real prevalence** — it must
never be used to make claims about real people, places, or programs, and the
app never presents output as an eligibility decision.

> **Status:** complete. Phase 1 = training + artifacts; Phase 2 = the FastAPI
> web app. 16 tests passing.

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

## Run the app

```bash
uvicorn app:app --reload
```

Then open <http://127.0.0.1:8000>. The form is prefilled with a typical
household, so you can submit it immediately.

Routes: `/` (form) · `POST /predict` (HTML result) · `POST /api/predict`
(JSON) · `/health` (artifact versions + schema hash).

```bash
pytest        # 14 tests
```

The app fails to start on purpose if an artifact is missing or a model's schema
hash does not match `feature_schema.json` — it never silently serves a stale
model. If startup complains, run `python train.py` and try again.

## Deploy (Render)

The repo ships with `render.yaml`, a `Procfile`, and a pinned Python version,
so a container host runs the app with **no code changes**.

1. Push to GitHub (already done for this repo).
2. On [render.com](https://render.com), sign in with GitHub, then
   **New + → Blueprint** and pick this repo. Render reads `render.yaml` and
   configures the build (`pip install -r requirements.txt`) and start command
   (`uvicorn app:app --host 0.0.0.0 --port $PORT`) automatically.
3. **Apply**. First build takes ~3–5 min (it installs scikit-learn/pandas).
   You get a public URL like `https://household-needs-check.onrender.com`.

> **Live app:** _add your Render URL here once it's deployed._

The small model artifacts (`models/*.joblib`, `feature_schema.json`,
`weights.yaml`) are committed so the deploy has everything it needs; the 30 MB
training dataset is **not** required at runtime and is excluded.

Free-tier note: the service spins down after ~15 min idle, so the first request
after a lull takes ~30–50 s to wake, then it's instant.

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

## How to read the results page

The results page has three boxes. None of them is a decision — read them as an
estimate and as starting points a person would review.

1. **Estimated food-insecurity risk** — one big percentage, e.g. "30%." That is
   the model's *estimated probability* that this household is food-insecure. Next
   to it is a plain-English band (lower / moderate / elevated / higher).
   - **The threshold slider.** A screening tool has to draw a line: at or above
     it, "flag for review"; below it, don't. Drag the slider and the "flagged /
     not flagged" text updates live. Slide it **left** and you flag more
     households (you catch more truly food-insecure ones, but also more false
     alarms). Slide it **right** and you flag fewer (fewer false alarms, but you
     miss more). That trade-off is called precision vs. recall. The shipped line
     is 50%.
   - **"What pushed this estimate up or down."** Each bar is one feature's
     contribution. Bars to the **right (warm)** pushed the estimate up; bars to
     the **left (cool)** pushed it down; longer = stronger. Read this as *"the
     model associates these answers with higher/lower risk,"* **not** as *"this
     caused it."* Association is not cause.

2. **Hardship score (0–100)** — this is **not** a machine-learning model. It is a
   transparent recipe: several need signals, each scored 0–1, each multiplied by
   a weight, added up. The bars show how many points each part contributed. You
   could reproduce the number by hand, and you can change the weights in
   `models/weights.yaml`.

3. **Suggested policy-priority segment** — one of five service buckets (e.g.
   "housing emergency"), with a confidence and the full five-way split. It is a
   *suggested starting point for review*, not an assignment.

At the bottom, "Computed from your answers" shows the three fields the app
worked out for you (housing-cost burden, income-to-poverty ratio, food-desert
flag) so nothing is hidden.

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
app.py         FastAPI routes + prediction core (importable by tests)
schema.py      schema-driven request validation + fail-loud hash checks
hardship.py    transparent hardship score + shared feature derivation
templates/     base, form, result, about, leakage (Jinja2)
static/        style.css, app.js (threshold slider)
tests/         pytest suite
models/        artifacts (weights.yaml is tracked; the rest regenerate)
notes/         derived.md — recovered generator rules
```

## Privacy

The app is **stateless**. Nothing you enter is persisted or logged; a result is
computed and discarded. Reload and it is gone. Every page carries a banner
saying so, and that this is a demonstration, not an eligibility determination.
