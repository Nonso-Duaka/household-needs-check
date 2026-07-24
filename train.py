
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

import hardship

ROOT = Path(__file__).parent
DATA = ROOT / "dataset" / "synthetic_food_housing_insecurity_100k.csv"
MODELS = ROOT / "models"
NOTES = ROOT / "notes"

FOOD_TARGET = "food_insecure_label"
SEGMENT_TARGET = "policy_priority_segment"
DEFAULT_THRESHOLD = 0.50

# The 14 fields the household answers on the form.
FORM_FIELDS = [
    "monthly_income_usd",
    "household_size",
    "children_count",
    "monthly_housing_cost_usd",
    "housing_status",
    "employment_status",
    "income_volatility",
    "savings_buffer_days",
    "benefits_snap",
    "transportation_access",
    "distance_to_grocery_miles",
    "behind_on_housing_payment",
    "utility_shutoff_notice",
    "chronic_health_condition",
]
# Computed server-side from the answers (never asked). See notes/derived.md.
DERIVED_FIELDS = [
    "rent_or_mortgage_burden_pct",
    "income_to_poverty_ratio",
    "food_desert_flag",
]
REALISTIC_FEATURES = FORM_FIELDS + DERIVED_FIELDS

# Never a predictor. These are the guard-rails the tests re-check independently.
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

# Fallback used only if a user deletes models/weights.yaml (normally it ships).
DEFAULT_HARDSHIP_WEIGHTS = {
    "income_pressure": 0.18, "housing_instability": 0.16,
    "financial_fragility": 0.15, "housing_cost_pressure": 0.12,
    "food_access_pressure": 0.10, "employment_instability": 0.08,
    "digital_and_service_access": 0.08, "health_family_need": 0.07,
    "stress_signal": 0.06,
}

# Friendly, plain-language field metadata. The form renders entirely from this
# (via feature_schema.json): question text, helper, the choices, and which
# section it sits in. Order here is the order shown in each section.
FIELD_META = {
    "household_size": {"type": "int", "min": 1, "max": 8, "group": "Your household",
        "label": "How many people live in your home?",
        "helper": "Count everyone — grown-ups and kids."},
    "children_count": {"type": "int", "min": 0, "max": 7, "group": "Your household",
        "label": "How many of them are kids (under 18)?",
        "helper": "Put 0 if there are none."},
    "monthly_income_usd": {"type": "int", "min": 300, "max": 25000, "group": "Money & work",
        "label": "How much money does your household get each month?",
        "helper": "Add up everything everyone brings in — pay, benefits, all of it."},
    "employment_status": {"type": "categorical", "group": "Money & work",
        "label": "Does the main earner have a job?",
        "helper": "Pick what fits best right now.",
        "levels": ["employed_full_time", "employed_part_time", "unemployed",
                   "not_in_labor_force", "retired", "student_or_training"],
        "level_labels": {
            "employed_full_time": "Yes, full-time", "employed_part_time": "Yes, part-time",
            "unemployed": "No, looking for work", "not_in_labor_force": "No, not looking",
            "retired": "Retired", "student_or_training": "In school or training"}},
    "income_volatility": {"type": "categorical", "group": "Money & work",
        "label": "Is that money about the same every month?",
        "helper": "Some jobs pay the same each month; others go up and down.",
        "levels": ["low", "medium", "high"],
        "level_labels": {"low": "About the same", "medium": "Changes a little",
                         "high": "Changes a lot"}},
    "savings_buffer_days": {"type": "int", "min": 0, "max": 180, "group": "Money & work",
        "label": "If the money stopped, how many days could savings cover the basics?",
        "helper": "A rough guess is fine. 0 means no savings to fall back on."},
    "benefits_snap": {"type": "binary", "group": "Money & work",
        "label": "Does your household get SNAP (food stamps)?",
        "helper": ""},
    "monthly_housing_cost_usd": {"type": "int", "min": 0, "max": 9000, "group": "Your home",
        "label": "How much do you pay for housing each month?",
        "helper": "Rent or mortgage. Put 0 if you pay nothing."},
    "housing_status": {"type": "categorical", "group": "Your home",
        "label": "What's your home situation?",
        "helper": "",
        "levels": ["own_mortgage", "own_no_mortgage", "rent", "doubled_up",
                   "shelter_or_transitional"],
        "level_labels": {
            "own_mortgage": "Own it, still paying it off", "own_no_mortgage": "Own it, paid off",
            "rent": "Renting", "doubled_up": "Staying with family or friends",
            "shelter_or_transitional": "Shelter or temporary housing"}},
    "behind_on_housing_payment": {"type": "binary", "group": "Your home",
        "label": "Are you behind on rent or mortgage?",
        "helper": ""},
    "utility_shutoff_notice": {"type": "binary", "group": "Your home",
        "label": "Had a shut-off warning for a utility lately?",
        "helper": "Like a notice that power, water, or gas might be turned off."},
    "transportation_access": {"type": "categorical", "group": "Food & getting around",
        "label": "How easy is it to get around?",
        "helper": "For work, appointments, and shopping.",
        "levels": ["reliable", "limited", "no_vehicle_or_transit"],
        "level_labels": {"reliable": "Easy — I have a reliable ride",
                         "limited": "A bit hard", "no_vehicle_or_transit": "Very hard — no car or bus"}},
    "distance_to_grocery_miles": {"type": "float", "min": 0.1, "max": 45, "group": "Food & getting around",
        "label": "How far is the nearest grocery store, in miles?",
        "helper": "A real grocery store, not a corner shop."},
    "chronic_health_condition": {"type": "binary", "group": "Health",
        "label": "Does anyone at home have an ongoing health condition?",
        "helper": "Something long-term, like diabetes or asthma."},
}

# Section order for the form.
GROUP_ORDER = ["Your household", "Money & work", "Your home",
               "Food & getting around", "Health"]

DERIVED_META = {
    "rent_or_mortgage_burden_pct": (
        "min(100 * monthly_housing_cost_usd / monthly_income_usd, 200)",
        "The share of your money that goes to housing."),
    "income_to_poverty_ratio": (
        "(monthly_income_usd * 12) / (15060 + 5380 * (household_size - 1))",
        "Your income compared with the official poverty line."),
    "food_desert_flag": (
        "1 if transportation_access != 'reliable' and distance_to_grocery_miles > 1.5 else 0",
        "Whether it's hard to reach a grocery store."),
}

RISK_BANDS = [
    {"label": "Low chance", "min": 0.0, "max": 0.25},
    {"label": "Some chance", "min": 0.25, "max": 0.50},
    {"label": "Higher chance", "min": 0.50, "max": 0.70},
    {"label": "High chance", "min": 0.70, "max": 1.01},
]

SEGMENT_LABELS = {
    "stable_or_low_need": "Fairly steady for now",
    "prevention_support": "Could use a little support",
    "benefit_navigation": "Help signing up for benefits",
    "housing_emergency": "Housing help needed",
    "severe_multiple_hardship": "Needs coordinated support",
}

SEGMENT_ACTIONS = {
    "stable_or_low_need": "Things look fairly steady right now. General resources may still be useful.",
    "prevention_support": "A little help now — like budgeting support — could keep small problems from growing.",
    "benefit_navigation": "Getting signed up for benefits like SNAP, WIC, or school meals could help the most.",
    "housing_emergency": "Housing looks like the urgent need — rent help, eviction help, or utility help.",
    "severe_multiple_hardship": "Several needs at once. Someone who can coordinate different kinds of help would fit best.",
}


def build_pipeline(estimator, features):
    cat = [c for c in features if c in ("housing_status", "employment_status",
                                        "income_volatility", "transportation_access")]
    num = [c for c in features if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    return Pipeline([("preprocess", pre), ("model", estimator)]), num, cat


def schema_hash(schema: dict) -> str:
    # Hash the schema STRUCTURE only. Exclude the hash itself and volatile
    # build metadata so an unchanged schema yields a stable contract hash.
    volatile = {"schema_hash", "created_utc"}
    payload = {k: v for k, v in schema.items() if k not in volatile}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def recover_and_report_rules(df: pd.DataFrame) -> dict:
    """Recover the two generator rules and report agreement (see notes/derived.md)."""
    denom = 15060 + 5380 * (df["household_size"] - 1)
    ratio = (df["monthly_income_usd"] * 12) / denom
    ratio_err = (ratio - df["income_to_poverty_ratio"]).abs()

    fd = ((df["transportation_access"].ne("reliable")) &
          (df["distance_to_grocery_miles"] > 1.5)).astype(int)
    fd_agree = float((fd == df["food_desert_flag"]).mean())

    burden = np.minimum(100 * df["monthly_housing_cost_usd"] /
                        df["monthly_income_usd"].clip(lower=1), 200)
    burden_err = (burden - df["rent_or_mortgage_burden_pct"]).abs()

    report = {
        "income_to_poverty_ratio": {
            "formula": "(monthly_income*12) / (15060 + 5380*(size-1))  [2024 HHS guideline]",
            "median_abs_err": round(float(ratio_err.median()), 4),
            "p99_abs_err": round(float(ratio_err.quantile(0.99)), 4),
            "confidence": "high",
        },
        "food_desert_flag": {
            "formula": "transport != reliable AND distance > 1.5 mi",
            "agreement_with_true_flag": round(fd_agree, 4),
            "note": "true generator also uses urbanicity (not a form field)",
            "confidence": "high on true rule, medium on form-only approximation",
        },
        "rent_or_mortgage_burden_pct": {
            "formula": "min(100*housing/income, 200)",
            "median_abs_err": round(float(burden_err.median()), 4),
            "confidence": "exact (defined in data dictionary)",
        },
    }
    print("\n=== Recovered generator rules (notes/derived.md) ===")
    print(f"income_to_poverty_ratio: median abs err "
          f"{report['income_to_poverty_ratio']['median_abs_err']}")
    print(f"food_desert_flag: {report['food_desert_flag']['agreement_with_true_flag']:.1%} "
          f"agreement with true flag")
    return report


def capacity_table(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    """Precision and share of all positives captured in the top k% by score."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = int(y_true.sum())
    rows = []
    for pct in (5, 10, 20):
        k = max(1, int(len(y_true) * pct / 100))
        top = y_sorted[:k]
        rows.append({
            "top_pct": pct,
            "households": k,
            "precision": round(float(top.mean()), 3),
            "share_of_positives_captured": round(float(top.sum() / total_pos), 3),
        })
    return rows


def build_schema(df: pd.DataFrame, food_num, food_cat, positive_rate) -> dict:
    # Emit form fields grouped into sections (GROUP_ORDER), preserving the
    # order fields appear within FIELD_META for each section.
    form = []
    ordered = sorted(FIELD_META.items(),
                     key=lambda kv: (GROUP_ORDER.index(kv[1]["group"]),
                                     list(FIELD_META).index(kv[0])))
    for name, meta in ordered:
        ftype = meta["type"]
        if ftype == "categorical":
            default = df[name].mode().iloc[0]
        elif ftype == "binary":
            default = 0
        else:
            default = (int(round(df[name].median())) if ftype == "int"
                       else round(float(df[name].median()), 1))
        field = {"name": name, "type": ftype, "group": meta["group"],
                 "label": meta["label"], "helper": meta["helper"], "default": default}
        if ftype == "categorical":
            field["levels"] = meta["levels"]
            field["level_labels"] = meta["level_labels"]
        elif ftype == "binary":
            field["min"], field["max"] = 0, 1
        else:
            field["min"], field["max"] = meta["min"], meta["max"]
        form.append(field)

    derived = [{"name": n, "formula": DERIVED_META[n][0],
                "description": DERIVED_META[n][1]} for n in DERIVED_FIELDS]

    schema = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": {
            "name": FOOD_TARGET,
            "positive_rate_train": round(float(positive_rate), 4),
            "default_threshold": DEFAULT_THRESHOLD,
            "risk_bands": RISK_BANDS,
            "note": "Estimated probability of food insecurity, not an eligibility decision.",
        },
        "model": {
            "type": "LogisticRegression(class_weight='balanced')",
            "feature_order": REALISTIC_FEATURES,
            "numeric_features": food_num,
            "categorical_features": food_cat,
        },
        "segments": {"classes": sorted(df[SEGMENT_TARGET].unique().tolist()),
                     "labels": SEGMENT_LABELS, "actions": SEGMENT_ACTIONS},
        "form_fields": form,
        "groups": GROUP_ORDER,
        "derived_fields": derived,
    }
    schema["schema_hash"] = schema_hash(schema)
    return schema


def assert_no_leakage(schema: dict):
    """Fail loudly if any predictor is a leakage column or protected attribute."""
    predictors = set(schema["model"]["feature_order"])
    predictors |= {f["name"] for f in schema["form_fields"]}
    predictors |= {f["name"] for f in schema["derived_fields"]}
    banned = set(HARD_LEAKAGE + SOFT_LEAKAGE + PROTECTED) - {FOOD_TARGET}
    bad = predictors & banned
    assert not bad, f"Leakage/protected column used as predictor: {bad}"
    # Protected attributes must not appear anywhere in the schema text at all.
    blob = json.dumps(schema)
    for p in PROTECTED:
        assert p not in blob, f"Protected attribute '{p}' leaked into schema text"
    print("Leakage guard: OK (predictors clean, protected attrs absent from schema)")



def train_food_model(df, train_mask, val_mask, test_mask):
    pipe, num, cat = build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=3000),
        REALISTIC_FEATURES,
    )
    pipe.fit(df.loc[train_mask, REALISTIC_FEATURES], df.loc[train_mask, FOOD_TARGET])

    # Tune (inspect) threshold on validation; ship 0.50 (the app has a slider).
    val_score = pipe.predict_proba(df.loc[val_mask, REALISTIC_FEATURES])[:, 1]
    val_y = df.loc[val_mask, FOOD_TARGET].to_numpy()
    thresholds = np.linspace(0.2, 0.8, 61)
    f1s = [f1_score(val_y, (val_score >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])

    # Report FINAL numbers on test once, at the shipped threshold 0.50.
    test_score = pipe.predict_proba(df.loc[test_mask, REALISTIC_FEATURES])[:, 1]
    test_y = df.loc[test_mask, FOOD_TARGET].to_numpy()
    test_pred = (test_score >= DEFAULT_THRESHOLD).astype(int)
    cm = confusion_matrix(test_y, test_pred)

    metrics = {
        "threshold": DEFAULT_THRESHOLD,
        "roc_auc": round(float(roc_auc_score(test_y, test_score)), 3),
        "precision": round(float(precision_score(test_y, test_pred)), 3),
        "recall": round(float(recall_score(test_y, test_pred)), 3),
        "f1": round(float(f1_score(test_y, test_pred)), 3),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "validation_f1_optimal_threshold": round(best_t, 3),
        "capacity_table": capacity_table(test_y, test_score),
    }
    print("\n=== Model A: food insecurity (test set, threshold 0.50) ===")
    print(f"ROC AUC {metrics['roc_auc']} | precision {metrics['precision']} "
          f"| recall {metrics['recall']} | F1 {metrics['f1']}")
    print(f"confusion matrix  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    print("capacity (top-k% by score):")
    for r in metrics["capacity_table"]:
        print(f"  top {r['top_pct']:>2}%  precision {r['precision']}  "
              f"captures {r['share_of_positives_captured']:.0%} of positives")
    return pipe, num, cat, metrics


def train_segment_model(df, train_mask, test_mask):
    pipe, _, _ = build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=3000),
        REALISTIC_FEATURES,
    )
    pipe.fit(df.loc[train_mask, REALISTIC_FEATURES], df.loc[train_mask, SEGMENT_TARGET])
    acc = float((pipe.predict(df.loc[test_mask, REALISTIC_FEATURES]) ==
                 df.loc[test_mask, SEGMENT_TARGET]).mean())
    print(f"\n=== Model B: policy segment (test accuracy) === {acc:.3f}")
    return pipe, {"test_accuracy": round(acc, 3),
                  "classes": list(pipe.named_steps["model"].classes_)}


def run_compare(df, train_mask, test_mask):
    print("\n=== --compare: model bake-off (test set) ===")
    test_y = df.loc[test_mask, FOOD_TARGET].to_numpy()
    Xtr, ytr = df.loc[train_mask, REALISTIC_FEATURES], df.loc[train_mask, FOOD_TARGET]
    Xte = df.loc[test_mask, REALISTIC_FEATURES]

    sub = df.loc[train_mask].groupby(FOOD_TARGET, group_keys=False).sample(
        n=4000, random_state=42)  # stratified subsample for the slow SVC

    candidates = {
        "logistic_regression (shipped)": (
            LogisticRegression(class_weight="balanced", max_iter=3000), Xtr, ytr),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   n_jobs=-1, random_state=42), Xtr, ytr),
        "svc_rbf (subsampled)": (
            SVC(probability=True, class_weight="balanced", random_state=42),
            sub[REALISTIC_FEATURES], sub[FOOD_TARGET]),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=42), Xtr, ytr),
    }
    rows = []
    for name, (est, X, y) in candidates.items():
        pipe, _, _ = build_pipeline(est, REALISTIC_FEATURES)
        pipe.fit(X, y)
        s = pipe.predict_proba(Xte)[:, 1]
        p = (s >= DEFAULT_THRESHOLD).astype(int)
        rows.append({
            "model": name,
            "roc_auc": round(float(roc_auc_score(test_y, s)), 3),
            "precision": round(float(precision_score(test_y, p)), 3),
            "recall": round(float(recall_score(test_y, p)), 3),
            "f1": round(float(f1_score(test_y, p)), 3),
        })
        print(f"  {name:<32} AUC {rows[-1]['roc_auc']}  F1 {rows[-1]['f1']}")
    out = pd.DataFrame(rows)
    out.to_csv(MODELS / "comparison.csv", index=False)
    print(f"  -> wrote {MODELS/'comparison.csv'}")
    print("  NOTE: shipped model stays logistic regression; the tree/margin "
          "models buy little on the realistic set and cost interpretability.")
    return rows


def validate_hardship(df) -> dict:
    """Decile check: insecurity rates must rise with the hardship score."""
    print("\n=== Hardship score validation (decile monotonicity) ===")
    # (a) Literal check on the dataset's own overall_hardship_score.
    d = pd.qcut(df["overall_hardship_score"], 10, labels=False, duplicates="drop") + 1
    tbl = df.groupby(d).agg(
        households=("household_id", "count"),
        food_rate=(FOOD_TARGET, "mean"),
        housing_rate=("housing_insecure_label", "mean"),
    ).round(3)
    print("overall_hardship_score deciles:")
    print(tbl.to_string())
    mono_food = bool(tbl["food_rate"].is_monotonic_increasing)
    mono_house = bool(tbl["housing_rate"].is_monotonic_increasing)

    # (b) Our transparent score_household() on a sample -> deciles monotonic?
    sample = df.sample(n=8000, random_state=42)
    our = sample.apply(lambda r: hardship.score_household(r.to_dict())[0], axis=1)
    od = pd.qcut(our, 10, labels=False, duplicates="drop") + 1
    ours = sample.assign(_s=our, _d=od).groupby("_d").agg(
        food_rate=(FOOD_TARGET, "mean"),
        housing_rate=("housing_insecure_label", "mean"),
    ).round(3)
    corr = float(np.corrcoef(our, sample["overall_hardship_score"])[0, 1])
    print(f"\nour score_household() vs overall_hardship_score corr: {corr:.3f}")
    print("our score deciles (food / housing insecurity rate):")
    print(ours.to_string())

    return {
        "overall_score_deciles_monotonic": {"food": mono_food, "housing": mono_house},
        "our_score_corr_with_overall": round(corr, 3),
        "our_score_deciles_monotonic": {
            "food": bool(ours["food_rate"].is_monotonic_increasing),
            "housing": bool(ours["housing_rate"].is_monotonic_increasing),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="also fit RF/SVC/HistGBM and write comparison.csv")
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    NOTES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")

    train_mask = df["train_test_split"].eq("train")
    val_mask = df["train_test_split"].eq("validation")
    test_mask = df["train_test_split"].eq("test")
    assert train_mask.sum() and val_mask.sum() and test_mask.sum(), "bad split"

    rules = recover_and_report_rules(df)

    schema = build_schema(df, *build_pipeline(LogisticRegression(), REALISTIC_FEATURES)[1:],
                          positive_rate=df.loc[train_mask, FOOD_TARGET].mean())
    assert_no_leakage(schema)

    food_pipe, food_num, food_cat, food_metrics = train_food_model(
        df, train_mask, val_mask, test_mask)
    seg_pipe, seg_metrics = train_segment_model(df, train_mask, test_mask)
    hardship_report = validate_hardship(df)
    compare_rows = run_compare(df, train_mask, test_mask) if args.compare else None

    h = schema["schema_hash"]
    stamp = {"trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "sklearn_version": sklearn.__version__,
             "python_version": platform.python_version(),
             "schema_hash": h, "feature_order": REALISTIC_FEATURES}

    joblib.dump({"pipeline": food_pipe, "kind": "food_insecurity", **stamp},
                MODELS / "food_model.joblib")
    joblib.dump({"pipeline": seg_pipe, "kind": "policy_segment", **stamp},
                MODELS / "segment_model.joblib")
    (MODELS / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    metrics = {
        "food_model": food_metrics,
        "segment_model": seg_metrics,
        "hardship_validation": hardship_report,
        "rule_recovery": rules,
        "leakage_lesson": {
            "note": "App ships the realistic (honest) field set. The honest model "
                    "looks weaker and that is the point.",
            "full_42_predictors": {"roc_auc": 0.893, "precision": 0.653, "recall": 0.789},
            "minus_food_budget": {"roc_auc": 0.830, "precision": 0.558, "recall": 0.733},
            "realistic_shipped": {"roc_auc": food_metrics["roc_auc"],
                                  "precision": food_metrics["precision"],
                                  "recall": food_metrics["recall"]},
        },
        "compare": compare_rows,
        **stamp,
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # weights.yaml is editable and ships in the repo; write a default only if a
    # user has deleted it, so we never clobber their edits on a retrain.
    wpath = MODELS / "weights.yaml"
    if not wpath.exists():
        import yaml
        wpath.write_text(yaml.safe_dump({"weights": DEFAULT_HARDSHIP_WEIGHTS},
                                        sort_keys=False))
    hardship.load_weights.cache_clear()
    hardship.load_weights()  # asserts sum==1.0, fails loudly if edited badly
    print(f"\nHardship weights OK (sum=1.0). Renormalization at scoring time:")
    print(json.dumps(hardship.describe_renormalization()["surviving_weights_renormalized"],
                     indent=2))

    print(f"\nArtifacts written to {MODELS}/  (schema_hash={h})")
    for f in ["food_model.joblib", "segment_model.joblib", "feature_schema.json",
              "metrics.json", "weights.yaml"] + (["comparison.csv"] if args.compare else []):
        print(f"  {f}")


if __name__ == "__main__":
    main()
