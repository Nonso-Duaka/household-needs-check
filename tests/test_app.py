"""Tests for the household screening app.

Run:  pytest -q
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as A
import hardship
import schema as S

client = TestClient(A.app)
ROOT = Path(__file__).resolve().parents[1]

# Guard-rails, defined here independently of the app so the test is a real check.
HARD_LEAKAGE = [
    "household_id", "survey_year", "train_test_split",
    "food_security_status", "food_insecure_label",
    "pantry_use_last_30d", "meals_skipped_last_30d",
    "months_food_shortage_last_year",
    "overall_hardship_score", "policy_priority_segment",
]
SOFT_LEAKAGE = ["food_budget_monthly_usd", "mental_health_stress_score"]
PROTECTED = ["race_ethnicity", "immigrant_household", "primary_language",
             "disability_present"]

LOW_NEED = {
    "monthly_income_usd": 12000, "household_size": 2, "children_count": 0,
    "monthly_housing_cost_usd": 1500, "housing_status": "own_mortgage",
    "employment_status": "employed_full_time", "income_volatility": "low",
    "savings_buffer_days": 150, "benefits_snap": 0,
    "transportation_access": "reliable", "distance_to_grocery_miles": 1.0,
    "behind_on_housing_payment": 0, "utility_shutoff_notice": 0,
    "chronic_health_condition": 0,
}
HIGH_NEED = {
    "monthly_income_usd": 900, "household_size": 5, "children_count": 3,
    "monthly_housing_cost_usd": 1400, "housing_status": "shelter_or_transitional",
    "employment_status": "unemployed", "income_volatility": "high",
    "savings_buffer_days": 0, "benefits_snap": 1,
    "transportation_access": "no_vehicle_or_transit", "distance_to_grocery_miles": 20.0,
    "behind_on_housing_payment": 1, "utility_shutoff_notice": 1,
    "chronic_health_condition": 1,
}


def defaults():
    return S.defaults()


# --- API accepts every schema field --------------------------------------
def test_every_schema_field_accepted():
    payload = defaults()
    assert set(payload) == {f["name"] for f in S.FORM_FIELDS}
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert {"food", "hardship", "segment", "derived"} <= body.keys()


# --- Out-of-range and missing -> 422, not 500 ----------------------------
def test_out_of_range_returns_422():
    bad = defaults()
    bad["household_size"] = 99  # max is 8
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 422


def test_negative_out_of_range_returns_422():
    bad = defaults()
    bad["monthly_income_usd"] = -5
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 422


def test_missing_field_returns_422():
    bad = defaults()
    del bad["monthly_income_usd"]
    r = client.post("/api/predict", json=bad)
    assert r.status_code == 422


# --- Unseen categorical level does not crash the encoder -----------------
def test_unseen_category_does_not_crash_encoder():
    # Bypass the API's schema layer and hit the model core directly, so the
    # OneHotEncoder(handle_unknown='ignore') is actually exercised.
    weird = dict(defaults())
    weird["housing_status"] = "houseboat_on_mars"  # never seen in training
    out = A.predict_food(weird)
    assert 0.0 <= out["probability"] <= 1.0


# --- Hardship weights sum to 1.0 -----------------------------------------
def test_hardship_weights_sum_to_one():
    w = hardship.load_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9


# --- High-need household outranks low-need on all three outputs -----------
def test_high_need_outranks_low_need():
    hi = A.score_all(HIGH_NEED)
    lo = A.score_all(LOW_NEED)
    assert hi["food"]["probability"] > lo["food"]["probability"]
    assert hi["hardship"]["score"] > lo["hardship"]["score"]
    # Segment: high-need should carry more probability mass on the severe end.
    def severe_mass(res):
        return sum(d["prob"] for d in res["segment"]["distribution"]
                   if d["segment"] in ("severe_multiple_hardship", "housing_emergency"))
    assert severe_mass(hi) > severe_mass(lo)


# --- Determinism ----------------------------------------------------------
def test_api_predict_is_deterministic():
    payload = defaults()
    a = client.post("/api/predict", json=payload).json()
    b = client.post("/api/predict", json=payload).json()
    assert a == b


# --- No leakage / protected attribute anywhere in the predictor schema ----
def test_schema_has_no_leakage_or_protected_predictors():
    schema = json.loads((ROOT / "models" / "feature_schema.json").read_text())
    predictors = set(schema["model"]["feature_order"])
    predictors |= {f["name"] for f in schema["form_fields"]}
    predictors |= {f["name"] for f in schema["derived_fields"]}
    banned = set(HARD_LEAKAGE + SOFT_LEAKAGE + PROTECTED) - {S.TARGET["name"]}
    assert not (predictors & banned), f"leakage predictor(s): {predictors & banned}"


def test_protected_attributes_absent_from_schema_text():
    blob = (ROOT / "models" / "feature_schema.json").read_text()
    for p in PROTECTED:
        assert p not in blob, f"protected attribute {p!r} leaked into schema"


def test_soft_leakage_absent_from_schema_text():
    blob = (ROOT / "models" / "feature_schema.json").read_text()
    for c in SOFT_LEAKAGE:
        assert c not in blob, f"soft-leakage column {c!r} leaked into schema"


# --- Pages render ---------------------------------------------------------
@pytest.mark.parametrize("path", ["/", "/health"])
def test_pages_load(path):
    assert client.get(path).status_code == 200


def test_health_reports_matching_schema_hash():
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["schema_hash"] == S.SCHEMA_HASH
    assert h["food_model"]["schema_hash"] == S.SCHEMA_HASH
    assert abs(h["hardship_weights_sum"] - 1.0) < 1e-9
