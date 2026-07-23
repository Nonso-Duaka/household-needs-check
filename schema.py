
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field, create_model

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
SCHEMA_PATH = MODELS / "feature_schema.json"


def compute_hash(schema: dict) -> str:
    """Must match train.py.schema_hash exactly (structure only)."""
    volatile = {"schema_hash", "created_utc"}
    payload = {k: v for k, v in schema.items() if k not in volatile}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        raise RuntimeError(
            f"Missing {SCHEMA_PATH}. Run `python train.py` before starting the app."
        )
    schema = json.loads(SCHEMA_PATH.read_text())
    recomputed = compute_hash(schema)
    if recomputed != schema.get("schema_hash"):
        raise RuntimeError(
            "feature_schema.json is corrupt: recorded schema_hash "
            f"{schema.get('schema_hash')} != recomputed {recomputed}. "
            "Retrain with `python train.py`."
        )
    return schema


SCHEMA = load_schema()
SCHEMA_HASH = SCHEMA["schema_hash"]
FORM_FIELDS: list[dict] = SCHEMA["form_fields"]
DERIVED_FIELDS: list[dict] = SCHEMA["derived_fields"]
FEATURE_ORDER: list[str] = SCHEMA["model"]["feature_order"]
TARGET = SCHEMA["target"]
DEFAULT_THRESHOLD: float = TARGET["default_threshold"]
RISK_BANDS = TARGET["risk_bands"]
SEGMENTS = SCHEMA["segments"]

# Readable label per field name (form + derived), for the results page.
FIELD_LABELS: dict[str, str] = {f["name"]: f["label"] for f in FORM_FIELDS}
FIELD_LABELS.update({f["name"]: f["description"] for f in DERIVED_FIELDS})


def risk_band(prob: float) -> str:
    for b in RISK_BANDS:
        if b["min"] <= prob < b["max"]:
            return b["label"]
    return RISK_BANDS[-1]["label"]


def _field_type(field: dict):
    """Map a schema field to a (python_type, pydantic Field) for validation.

    Numeric fields are range-bounded so out-of-range -> 422. Binary is 0/1.
    Categorical is accepted as a plain string on purpose: the HTML <select>
    constrains choices, and the model's OneHotEncoder(handle_unknown='ignore')
    tolerates an unseen level from the JSON API without crashing.
    """
    t = field["type"]
    if t == "categorical":
        return (str, Field(...))
    if t == "binary":
        return (int, Field(..., ge=0, le=1))
    if t == "int":
        return (int, Field(..., ge=field["min"], le=field["max"]))
    if t == "float":
        return (float, Field(..., ge=field["min"], le=field["max"]))
    raise ValueError(f"Unknown field type {t!r}")


def build_input_model():
    """Build the pydantic request model for the 14 form fields from the schema."""
    fields: dict[str, Any] = {f["name"]: _field_type(f) for f in FORM_FIELDS}
    return create_model("HouseholdInput", **fields)


HouseholdInput = build_input_model()


def defaults() -> dict[str, Any]:
    """Prefill values so the form can be submitted immediately."""
    return {f["name"]: f["default"] for f in FORM_FIELDS}
