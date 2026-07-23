"""Transparent household hardship score.

This is deliberately NOT a fitted model. It is a weighted sum of interpretable
components, each normalized to 0-1, so a first-year student can read exactly
why a score is what it is. Weights live in an editable YAML file
(``models/weights.yaml``); change a number, restart, and the score changes.

The formulas and starting weights come from the reference notebook
``04_hardship_scoring.ipynb``. Two things differ here because the web form
cannot ask everything the notebook had:

  * The ``stress_signal`` component (mental_health_stress_score) is soft-leakage
    and is not on the form -> its top-level weight is dropped.
  * The whole ``digital_and_service_access`` component (internet, childcare,
    health-insurance) is not on the form -> its top-level weight is dropped.
  * Within the surviving components, sub-signals that need non-form fields
    (eviction risk, debt-to-income, disability, senior-present) are dropped and
    the remaining sub-weights are renormalized.

After dropping, the surviving top-level weights are renormalized to sum to 1.0
so the score still spans 0-100. All of this is done explicitly and reported by
``describe_renormalization()`` rather than silently passing zeros. See
``notes/derived.md``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

WEIGHTS_PATH = Path(__file__).parent / "models" / "weights.yaml"

# --- constants for the two recovered generator rules (see notes/derived.md) ---
POVERTY_BASE = 15060          # 2024 HHS guideline, household of 1 (48 states)
POVERTY_PER_PERSON = 5380     # 2024 HHS guideline, per additional person
FOOD_DESERT_DISTANCE_MI = 1.5  # form-only approximation; ~97% agree with truth


def clip_scale(x: float, low: float, high: float, reverse: bool = False) -> float:
    """Clip x to [low, high] then scale to [0, 1]; optionally invert."""
    x = max(low, min(high, x))
    scaled = (x - low) / (high - low)
    return 1.0 - scaled if reverse else scaled


# ---------------------------------------------------------------------------
# Derived features (shared with the web app so there is ONE source of truth).
# ---------------------------------------------------------------------------
def derive_features(form: dict[str, Any]) -> dict[str, float]:
    """Compute the three server-side fields from raw form answers.

    Returns only the derived fields. Callers merge them into the form dict.
    """
    income = max(1.0, float(form["monthly_income_usd"]))
    housing = float(form["monthly_housing_cost_usd"])
    size = int(form["household_size"])
    distance = float(form["distance_to_grocery_miles"])
    transport = str(form["transportation_access"])

    burden = min(100.0 * housing / income, 200.0)
    denominator = POVERTY_BASE + POVERTY_PER_PERSON * (size - 1)
    ratio = (income * 12.0) / denominator
    food_desert = int(transport != "reliable" and distance > FOOD_DESERT_DISTANCE_MI)

    return {
        "rent_or_mortgage_burden_pct": round(burden, 2),
        "income_to_poverty_ratio": round(ratio, 4),
        "food_desert_flag": food_desert,
    }


def with_derived(form: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of form with derived fields added (existing keys win)."""
    merged = dict(form)
    for k, v in derive_features(form).items():
        merged.setdefault(k, v)
    return merged


# ---------------------------------------------------------------------------
# Weights.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_weights(path: str | None = None) -> dict[str, float]:
    p = Path(path) if path else WEIGHTS_PATH
    data = yaml.safe_load(p.read_text())
    weights = data["weights"]
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9, (
        f"Hardship weights in {p} must sum to 1.0, got {total:.6f}"
    )
    return dict(weights)


# Components the form cannot supply at all -> their top-level weight is dropped.
UNAVAILABLE_COMPONENTS = ("digital_and_service_access", "stress_signal")

# Employment instability lookup (from the reference notebook).
_EMPLOYMENT_RISK = {
    "employed_full_time": 0.0,
    "employed_part_time": 0.35,
    "gig_or_contract": 0.45,
    "unemployed": 1.0,
    "not_in_labor_force": 0.65,
    "retired": 0.25,
    # "student_or_training" is on the form but absent from the notebook map;
    # falls through to the 0.40 default below.
}
_VOLATILITY_RISK = {"low": 0.0, "medium": 0.5, "high": 1.0}


def _components(row: dict[str, Any]) -> dict[str, float]:
    """Each interpretable component, normalized to 0-1, from form+derived fields.

    Sub-weights are renormalized here over the form-available sub-signals only
    (the dropped ones are noted in describe_renormalization()).
    """
    children_present = 1 if int(row["children_count"]) > 0 else 0
    transport_not_reliable = int(str(row["transportation_access"]) != "reliable")

    comp: dict[str, float] = {}
    comp["income_pressure"] = clip_scale(
        float(row["income_to_poverty_ratio"]), 0.5, 3.0, reverse=True
    )
    comp["housing_cost_pressure"] = clip_scale(
        float(row["rent_or_mortgage_burden_pct"]), 20, 70
    )
    # housing_instability: drop eviction_or_foreclosure_risk (0.35), renorm 0.45/0.20
    comp["housing_instability"] = (
        0.6923 * int(row["behind_on_housing_payment"])
        + 0.3077 * int(row["utility_shutoff_notice"])
    )
    # financial_fragility: drop debt_to_income_pct (0.30), renorm 0.50/0.20
    comp["financial_fragility"] = (
        0.7143 * clip_scale(float(row["savings_buffer_days"]), 0, 60, reverse=True)
        + 0.2857 * _VOLATILITY_RISK.get(str(row["income_volatility"]), 0.5)
    )
    comp["employment_instability"] = _EMPLOYMENT_RISK.get(
        str(row["employment_status"]), 0.40
    )
    comp["food_access_pressure"] = (
        0.40 * clip_scale(float(row["distance_to_grocery_miles"]), 0, 15)
        + 0.35 * int(row["food_desert_flag"])
        + 0.25 * transport_not_reliable
    )
    # health_family_need: drop disability (0.35) and senior (0.20), renorm 0.25/0.20
    comp["health_family_need"] = (
        0.5556 * int(row["chronic_health_condition"])
        + 0.4444 * children_present
    )
    return comp


def _effective_weights(weights: dict[str, float]) -> dict[str, float]:
    """Drop unavailable components and renormalize the rest to sum to 1.0."""
    kept = {k: v for k, v in weights.items() if k not in UNAVAILABLE_COMPONENTS}
    total = sum(kept.values())
    return {k: v / total for k, v in kept.items()}


def score_household(
    household: dict[str, Any], weights: dict[str, float] | None = None
) -> tuple[float, list[dict[str, Any]]]:
    """Score one household 0-100 and return its component breakdown.

    ``household`` may be raw form answers; derived fields are added if missing.
    Returns ``(score, components)`` where components is a list of dicts sorted
    by contribution, each with: name, raw (0-1), weight, contribution (points).
    """
    if weights is None:
        weights = load_weights()
    row = with_derived(household)
    comp = _components(row)
    eff = _effective_weights(weights)

    breakdown = []
    score = 0.0
    for name, raw in comp.items():
        w = eff.get(name, 0.0)
        points = 100.0 * w * raw
        score += points
        breakdown.append(
            {
                "name": name,
                "raw": round(raw, 4),
                "weight": round(w, 4),
                "contribution": round(points, 2),
            }
        )
    breakdown.sort(key=lambda d: d["contribution"], reverse=True)
    return round(score, 1), breakdown


def describe_renormalization(weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Human-readable record of exactly what was dropped and renormalized."""
    if weights is None:
        weights = load_weights()
    eff = _effective_weights(weights)
    return {
        "dropped_components": {
            k: weights[k] for k in UNAVAILABLE_COMPONENTS if k in weights
        },
        "dropped_reason": {
            "digital_and_service_access": "internet/childcare/health-insurance not asked on form",
            "stress_signal": "mental_health_stress_score is soft-leakage and not asked",
        },
        "surviving_weights_original": {
            k: v for k, v in weights.items() if k not in UNAVAILABLE_COMPONENTS
        },
        "surviving_weights_renormalized": {k: round(v, 4) for k, v in eff.items()},
        "dropped_subsignals": {
            "housing_instability": "eviction_or_foreclosure_risk (0.35) dropped",
            "financial_fragility": "debt_to_income_pct (0.30) dropped",
            "health_family_need": "disability_present (0.35), senior_present (0.20) dropped",
        },
    }
