from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import hardship
import schema as S

ROOT = Path(__file__).parent
MODELS = ROOT / "models"

# Short, plain labels for the "what changed the estimate" bars.
SHORT_LABELS = {
    "monthly_income_usd": "Monthly income", "household_size": "People at home",
    "children_count": "Number of kids", "monthly_housing_cost_usd": "Housing cost",
    "housing_status": "Home situation", "employment_status": "Job situation",
    "income_volatility": "Steady income?", "savings_buffer_days": "Savings cushion",
    "benefits_snap": "Gets SNAP", "transportation_access": "Getting around",
    "distance_to_grocery_miles": "Distance to store",
    "behind_on_housing_payment": "Behind on housing",
    "utility_shutoff_notice": "Utility shut-off notice",
    "chronic_health_condition": "Health condition",
    "rent_or_mortgage_burden_pct": "Housing share of income",
    "income_to_poverty_ratio": "Income vs. poverty line",
    "food_desert_flag": "Far from a store",
}

# Plain labels for the hardship-score parts.
HARDSHIP_LABELS = {
    "income_pressure": "Not enough money coming in",
    "housing_cost_pressure": "Housing takes a big share",
    "housing_instability": "Behind on housing or utilities",
    "financial_fragility": "Little savings to fall back on",
    "employment_instability": "Job situation",
    "food_access_pressure": "Hard to reach a store",
    "health_family_need": "Health and family needs",
}

def _load_bundle(name: str) -> dict:
    path = MODELS / name
    if not path.exists():
        raise RuntimeError(f"Missing artifact {path}. Run `python train.py`.")
    bundle = joblib.load(path)
    if bundle.get("schema_hash") != S.SCHEMA_HASH:
        raise RuntimeError(
            f"{name} was trained against schema_hash {bundle.get('schema_hash')} "
            f"but feature_schema.json is {S.SCHEMA_HASH}. Retrain with "
            "`python train.py` so the model and the form agree."
        )
    return bundle


FOOD_BUNDLE = _load_bundle("food_model.joblib")
SEGMENT_BUNDLE = _load_bundle("segment_model.joblib")
FOOD_PIPE = FOOD_BUNDLE["pipeline"]
SEGMENT_PIPE = SEGMENT_BUNDLE["pipeline"]
# Touch the weights once so a broken weights.yaml fails at startup, not per-request.
hardship.load_weights()

app = FastAPI(title="Household Needs Screening (demo)")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


def feature_row(form: dict[str, Any]) -> dict[str, Any]:
    """Merge the 14 answers with the 3 server-derived fields (17 total)."""
    return hardship.with_derived(form)


def _model_frame(row: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{k: row[k] for k in S.FEATURE_ORDER}])


def food_contributions(X: pd.DataFrame) -> list[dict]:
    """Signed per-feature push = coefficient x standardized value.

    Aggregated back to the original field. Deliberately labeled 'what pushed
    this estimate up or down' — associational, never causal.
    """
    pre = FOOD_PIPE.named_steps["preprocess"]
    model = FOOD_PIPE.named_steps["model"]
    z = pre.transform(X)
    z = z.toarray()[0] if hasattr(z, "toarray") else z[0]
    coef = model.coef_[0]
    names = pre.get_feature_names_out()

    by_field: dict[str, float] = {}
    for name, contribution in zip(names, z * coef):
        stripped = name.split("__", 1)[1]
        field = next((f for f in S.FEATURE_ORDER if stripped.startswith(f)), stripped)
        by_field[field] = by_field.get(field, 0.0) + float(contribution)

    rows = [
        {"field": f, "label": SHORT_LABELS.get(f, f),
         "contribution": round(v, 4),
         "direction": "up" if v > 0 else "down"}
        for f, v in by_field.items()
    ]
    rows.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return rows


def predict_food(form: dict[str, Any]) -> dict[str, Any]:
    row = feature_row(form)
    X = _model_frame(row)
    prob = float(FOOD_PIPE.predict_proba(X)[0, 1])
    return {
        "probability": round(prob, 4),
        "threshold": S.DEFAULT_THRESHOLD,
        "flagged_at_default": bool(prob >= S.DEFAULT_THRESHOLD),
        "risk_band": S.risk_band(prob),
        "contributions": food_contributions(X),
    }


def predict_segment(form: dict[str, Any]) -> dict[str, Any]:
    row = feature_row(form)
    X = _model_frame(row)
    proba = SEGMENT_PIPE.predict_proba(X)[0]
    classes = list(SEGMENT_PIPE.named_steps["model"].classes_)
    labels = S.SEGMENTS.get("labels", {})
    dist = sorted(
        [{"segment": c, "label": labels.get(c, c.replace("_", " ")),
          "prob": round(float(p), 4)}
         for c, p in zip(classes, proba)],
        key=lambda d: d["prob"], reverse=True,
    )
    top = dist[0]["segment"]
    return {
        "segment": top,
        "label": labels.get(top, top.replace("_", " ")),
        "confidence": dist[0]["prob"],
        "distribution": dist,
        "recommended_action": S.SEGMENTS["actions"].get(top, ""),
    }


def score_all(form: dict[str, Any]) -> dict[str, Any]:
    """The full three-output payload for one household."""
    row = feature_row(form)
    hs_score, hs_components = hardship.score_household(form)
    for c in hs_components:
        c["label"] = HARDSHIP_LABELS.get(c["name"], c["name"].replace("_", " "))
    food = predict_food(form)
    food["max_abs"] = max((abs(c["contribution"]) for c in food["contributions"]),
                          default=1.0) or 1.0
    return {
        "food": food,
        "hardship": {
            "score": hs_score,
            "components": hs_components,
            "max_contribution": max((c["contribution"] for c in hs_components),
                                    default=1.0) or 1.0,
        },
        "segment": predict_segment(form),
        "derived": {k: row[k] for k in
                    ["rent_or_mortgage_burden_pct", "income_to_poverty_ratio",
                     "food_desert_flag"]},
    }



def grouped_fields():
    """Form fields bucketed into their sections, in schema order."""
    groups = S.SCHEMA.get("groups") or []
    sections = []
    for g in groups:
        fields = [f for f in S.FORM_FIELDS if f.get("group") == g]
        if fields:
            sections.append({"name": g, "fields": fields})
    # Any ungrouped fields (defensive) go in a trailing catch-all.
    leftover = [f for f in S.FORM_FIELDS if f.get("group") not in groups]
    if leftover:
        sections.append({"name": "More", "fields": leftover})
    return sections


@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse(request, "form.html", {
        "sections": grouped_fields(), "values": S.defaults(),
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict_html(request: Request):
    raw = dict(await request.form())
    # Coerce form strings to the schema's types, then validate via pydantic.
    typed: dict[str, Any] = {}
    for f in S.FORM_FIELDS:
        v = raw.get(f["name"])
        if v is None or v == "":
            continue
        try:
            typed[f["name"]] = (int(v) if f["type"] in ("int", "binary")
                                else float(v) if f["type"] == "float" else v)
        except ValueError:
            typed[f["name"]] = v
    try:
        clean = S.HouseholdInput(**typed).model_dump()
    except Exception as exc:  # re-render form with a message
        return templates.TemplateResponse(request, "form.html", {
            "sections": grouped_fields(), "values": {**S.defaults(), **typed},
            "error": "Please double-check your answers and try again.",
        }, status_code=422)
    result = score_all(clean)
    return templates.TemplateResponse(request, "result.html", {
        "r": result, "inputs": clean, "field_labels": S.FIELD_LABELS,
    })


@app.post("/api/predict")
def predict_api(payload: S.HouseholdInput):  # type: ignore[valid-type]
    """JSON in, JSON out. Validation errors return 422 automatically."""
    return JSONResponse(score_all(payload.model_dump()))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "schema_hash": S.SCHEMA_HASH,
        "food_model": {"trained_utc": FOOD_BUNDLE.get("trained_utc"),
                       "schema_hash": FOOD_BUNDLE.get("schema_hash")},
        "segment_model": {"trained_utc": SEGMENT_BUNDLE.get("trained_utc"),
                          "schema_hash": SEGMENT_BUNDLE.get("schema_hash")},
        "sklearn_version": sklearn.__version__,
        "hardship_weights_sum": round(sum(hardship.load_weights().values()), 6),
    }
