"""Deterministic, constrained single/combined intervention search."""
import itertools
import json

import numpy as np
import pandas as pd

from clinical_rules import (
    DIET_DAYS_MIN, DIET_DAYS_MAX, DIET_DAYS_TARGETS, NORMAL_DIET,
    PLAN_FIELDS, TOLERANCE, changed_plan_fields, counterfactual_violations,
)
from model_utils import format_probability, standardize_value, inverse_standardized_value

DIET_DAYS_ORIGINAL_MIN = int(DIET_DAYS_MIN)
DIET_DAYS_ORIGINAL_MAX = int(DIET_DAYS_MAX)

dietary_label_map = {
    "DietaryRestriction_1": "Fasting", "DietaryRestriction_2": "Low-residue Diet",
    "DietaryRestriction_3": "Liquid Diet", "DietaryRestriction_4": "Normal Diet",
}
laxative_label_map = {
    "LaxativeRegimen_1": "PEG 2L", "LaxativeRegimen_2": "PEG 3L", "LaxativeRegimen_3": "PEG 4L",
    "LaxativeRegimen_4": "Sodium Phosphate", "LaxativeRegimen_5": "Mannitol", "LaxativeRegimen_6": "Magnesium Sulfate",
}
interval_label_map = {
    "BPtoColonoscopyinterval_1": "<120 min", "BPtoColonoscopyinterval_2": "120–240 min",
    "BPtoColonoscopyinterval_3": "240–360 min", "BPtoColonoscopyinterval_4": "≥360 min",
}
binary_label_map = {
    "BPEducationModality": {0: "Written + graphic/video education", 1: "Oral or written education"},
    "SplitDose_BP": {0: "No split-dose bowel preparation", 1: "Split-dose bowel preparation"},
    "PreColonoscopyPhysicalActivity": {0: "No physical activity after laxative use", 1: "Physical activity after laxative use"},
}
GROUPS = {"DietaryRestriction": 4, "LaxativeRegimen": 6, "BPtoColonoscopyinterval": 4}
LABEL_MAPS = {"DietaryRestriction": dietary_label_map, "LaxativeRegimen": laxative_label_map,
              "BPtoColonoscopyinterval": interval_label_map}
FIELD_NAMES = {
    "DietaryRestrictionDays": "Dietary restriction duration", "DietaryRestriction": "Dietary restriction strategy",
    "LaxativeRegimen": "Laxative regimen", "BPtoColonoscopyinterval": "Interval to colonoscopy",
    "BPEducationModality": "Bowel preparation education", "SplitDose_BP": "Split-dose bowel preparation",
    "PreColonoscopyPhysicalActivity": "Physical activity after laxative use",
}
RESULT_COLUMNS = [
    "ID", "Category", "Intervention Type", "Counterfactual Recommendation", "Changed Features",
    "Baseline Measure", "Post-Intervention Measure", "Number of Changed Factors", "Counterfactual Plan",
    "Original Predicted Probability", "Post-counterfactual Prediction Probability", "Absolute Risk Reduction",
    "Relative Risk Reduction", "Risk Decreasing", "Below Intervention Threshold", "Interpretation",
]


def _predict_probabilities(model, frame, feature_order):
    values = np.asarray(model.predict_proba(frame.loc[:, feature_order]), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive = np.flatnonzero(classes == 1)
    if values.shape != (len(frame), 2) or len(positive) != 1:
        raise ValueError("Expected a binary model with class 1 indicating poor bowel preparation.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("The model returned invalid probabilities.")
    return values[:, positive[0]]


def predict_single_risk(model, row, feature_order):
    return float(_predict_probabilities(model, pd.DataFrame([row]), feature_order)[0])


def set_onehot_group(row, group, target_col):
    row = row.copy()
    for col in group:
        if col in row.index: row[col] = 0
    if target_col in row.index: row[target_col] = 1
    return row


def decode_onehot_group(row, group, label_map):
    available = [col for col in group if col in row.index]
    active = [col for col in available if np.isclose(float(row[col]), 1., rtol=0, atol=TOLERANCE)]
    return label_map.get(active[0], active[0]) if len(active) == 1 else None


def get_binary_label(feature, value):
    number = int(round(float(value)))
    return binary_label_map.get(feature, {}).get(number, number)


def set_scalar_value(row, col, value):
    row = row.copy(); row[col] = value
    return row


def get_diet_days(row, scaler):
    if "DietaryRestrictionDays" not in row.index: return None
    return round(float(inverse_standardized_value(scaler, "DietaryRestrictionDays", row["DietaryRestrictionDays"])), 10)


def set_diet_days(row, scaler, target_days):
    row = row.copy()
    row["DietaryRestrictionDays"] = standardize_value(scaler, "DietaryRestrictionDays", target_days)
    return row


def postprocess_row(row, feature_order, scaler=None):
    row = row.copy()
    if scaler is not None and "DietaryRestriction_4" in row.index and np.isclose(row["DietaryRestriction_4"], 1.):
        row = set_diet_days(row, scaler, 0.)
    return row.loc[feature_order]


def decode_plan(row, feature_order, scaler):
    result = {"DietaryRestrictionDays": get_diet_days(row, scaler)}
    for field in ["BPEducationModality", "SplitDose_BP", "PreColonoscopyPhysicalActivity"]:
        if field not in row.index or not np.isclose(float(row[field]), [0., 1.], rtol=0, atol=TOLERANCE).any():
            raise ValueError(f"{field} must be encoded as 0/1.")
        result[field] = int(round(float(row[field])))
    for field, count in GROUPS.items():
        group = [f"{field}_{k}" for k in range(1, count+1)]
        if not set(group) <= set(feature_order):
            raise ValueError(f"Missing one-hot columns for {field}.")
        vector = row[group].to_numpy(float)
        binary = np.isclose(vector, 0., rtol=0, atol=TOLERANCE) | np.isclose(vector, 1., rtol=0, atol=TOLERANCE)
        if not binary.all():
            raise ValueError(f"Invalid one-hot encoding for {field}.")
        active = np.flatnonzero(np.isclose(vector, 1., rtol=0, atol=TOLERANCE))
        if len(active) != 1 or not np.isclose(vector.sum(), 1., rtol=0, atol=TOLERANCE):
            raise ValueError(f"Exactly one category must be active for {field}.")
        result[field] = int(active[0]+1)
    return result


def is_fasting_over_1_day(row, feature_order, scaler):
    plan = decode_plan(row, feature_order, scaler)
    return plan["DietaryRestriction"] == 1 and plan["DietaryRestrictionDays"] >= 1.0-TOLERANCE


def classify_suggestion(prob, intervention_threshold):
    return "Risk decreased below intervention threshold" if prob < intervention_threshold else "Risk decreased but remained above intervention threshold"


def _value_label(field, value):
    if field == "DietaryRestrictionDays": return f"{value:g} days"
    if field in binary_label_map: return get_binary_label(field, value)
    return LABEL_MAPS[field][f"{field}_{int(value)}"]


def _describe(original, candidate, feature_name_map):
    fields = changed_plan_fields(original, candidate)
    names = [feature_name_map.get(f, FIELD_NAMES[f]) for f in fields]
    before = [str(_value_label(f, original[f])) for f in fields]
    after = [str(_value_label(f, candidate[f])) for f in fields]
    return (" + ".join(f"{name}: {a} → {b}" for name, a, b in zip(names, before, after)),
            " + ".join(f"{name}: {value}" for name, value in zip(names, before)),
            " + ".join(f"{name}: {value}" for name, value in zip(names, after)))


def generate_single_intervention_candidates(original_row, feature_order, scaler, feature_name_map,
                                              exclude_fasting_over_1_day=True):
    """Atomic search actions; final feasibility is checked after combination.

    Duration/dose reductions remain in the search pool so they can participate
    in allowed combined changes, but are never returned as standalone advice.
    """
    original = decode_plan(original_row, feature_order, scaler)
    actions = []
    def add(field, target, function):
        actions.append({"Action Group": field, "Intervention Type": FIELD_NAMES[field],
                        "Changed Features": field, "Target": target, "apply": function})
    for days in DIET_DAYS_TARGETS:
        if abs(days-original["DietaryRestrictionDays"]) > TOLERANCE:
            add("DietaryRestrictionDays", days, lambda row, d=days: set_diet_days(row, scaler, d))
    for field in ["BPEducationModality", "SplitDose_BP", "PreColonoscopyPhysicalActivity"]:
        target = 1-original[field]
        if field in {"SplitDose_BP", "PreColonoscopyPhysicalActivity"} and original[field] == 1:
            continue
        add(field, target, lambda row, f=field, v=target: set_scalar_value(row, f, v))
    for field, count in GROUPS.items():
        group = [f"{field}_{k}" for k in range(1, count+1)]
        for category in range(1, count+1):
            if category == original[field]: continue
            add(field, category, lambda row, g=group, t=f"{field}_{category}": set_onehot_group(row, g, t))
    return actions


def _evaluate(model, patient_model_df, feature_order, scaler, feature_name_map,
              intervention_threshold, min_absolute_reduction, modes, exclude_fasting_over_1_day):
    empty = pd.DataFrame(columns=RESULT_COLUMNS)
    if not 0 < intervention_threshold < 1 or not np.isfinite(min_absolute_reduction) or min_absolute_reduction < 0:
        raise ValueError("Invalid intervention threshold or minimum risk reduction.")
    original_row = patient_model_df.iloc[0].loc[feature_order].copy()
    if not np.isfinite(original_row.to_numpy(float)).all():
        raise ValueError("Patient inputs must be complete and finite.")
    original = decode_plan(original_row, feature_order, scaler)
    if not DIET_DAYS_MIN-TOLERANCE <= original["DietaryRestrictionDays"] <= DIET_DAYS_MAX+TOLERANCE:
        raise ValueError("Dietary restriction duration must be between 0 and 2 days.")
    if original["DietaryRestriction"] == NORMAL_DIET and abs(original["DietaryRestrictionDays"]) > TOLERANCE:
        raise ValueError("Normal diet requires zero dietary restriction days.")
    original_prob = predict_single_risk(model, original_row, feature_order)
    if original_prob < intervention_threshold:
        return empty.copy(), empty.copy()
    actions = generate_single_intervention_candidates(original_row, feature_order, scaler, feature_name_map,
                                                      exclude_fasting_over_1_day)
    combinations = []
    if "single" in modes: combinations.extend((action,) for action in actions)
    if "combined" in modes:
        combinations.extend((a, b) for a, b in itertools.combinations(actions, 2) if a["Action Group"] != b["Action Group"])
    candidates = {}
    for combination in combinations:
        row = original_row.copy()
        for action in combination: row = action["apply"](row)
        row = postprocess_row(row, feature_order, scaler)
        plan = decode_plan(row, feature_order, scaler)
        fields = changed_plan_fields(original, plan)
        if not fields or counterfactual_violations(original, plan, exclude_fasting_over_1_day=exclude_fasting_over_1_day):
            continue
        if len(combination) > 1 and len(fields) < 2:
            continue
        identity = tuple(plan[f] for f in PLAN_FIELDS)
        if identity not in candidates or len(combination) < candidates[identity][2]:
            candidates[identity] = (row, plan, len(combination))
    if not candidates: return empty.copy(), empty.copy()
    items = list(candidates.values())
    matrix = pd.DataFrame([item[0] for item in items]).loc[:, feature_order]
    probabilities = _predict_probabilities(model, matrix, feature_order)
    records = []
    for index, ((_, plan, action_count), prob) in enumerate(zip(items, probabilities), 1):
        fields = changed_plan_fields(original, plan)
        recommendation, before, after = _describe(original, plan, feature_name_map)
        reduction = original_prob-float(prob)
        decreasing = reduction > max(min_absolute_reduction, 1e-12)
        category = "Single intervention" if len(fields) == 1 else "Combined intervention"
        records.append({"ID": index, "Category": category,
            "Intervention Type": FIELD_NAMES[fields[0]] if len(fields) == 1 else "Combined intervention",
            "Counterfactual Recommendation": recommendation, "Changed Features": ",".join(fields),
            "Baseline Measure": before, "Post-Intervention Measure": after,
            "Number of Changed Factors": len(fields), "Counterfactual Plan": json.dumps(plan, sort_keys=True),
            "Original Predicted Probability": original_prob, "Post-counterfactual Prediction Probability": float(prob),
            "Absolute Risk Reduction": reduction, "Relative Risk Reduction": reduction/original_prob if original_prob else np.nan,
            "Risk Decreasing": decreasing, "Below Intervention Threshold": bool(prob < intervention_threshold),
            "Interpretation": classify_suggestion(prob, intervention_threshold) if decreasing else "Risk did not decrease"})
    all_df = pd.DataFrame(records, columns=RESULT_COLUMNS).sort_values(
        ["Post-counterfactual Prediction Probability", "Number of Changed Factors", "ID"], kind="stable").reset_index(drop=True)
    return all_df.loc[all_df["Risk Decreasing"]].reset_index(drop=True), all_df


def evaluate_single_interventions(model, patient_model_df, patient_raw_df, feature_order, scaler, feature_name_map,
                                 intervention_threshold, min_absolute_reduction=0., exclude_fasting_over_1_day=True):
    return _evaluate(model, patient_model_df, feature_order, scaler, feature_name_map,
                     intervention_threshold, min_absolute_reduction, {"single"}, exclude_fasting_over_1_day)


def evaluate_pairwise_interventions(model, patient_model_df, patient_raw_df, feature_order, scaler, feature_name_map,
                                   intervention_threshold, min_absolute_reduction=0., exclude_fasting_over_1_day=True):
    return _evaluate(model, patient_model_df, feature_order, scaler, feature_name_map,
                     intervention_threshold, min_absolute_reduction, {"combined"}, exclude_fasting_over_1_day)


def scan_risk_decreasing_measures(model, patient_model_df, patient_raw_df, feature_order, scaler, feature_name_map,
                                 intervention_threshold=.135, min_absolute_reduction=0., evaluate_pairwise=True,
                                 exclude_fasting_over_1_day=True):
    modes = {"single", "combined"} if evaluate_pairwise else {"single"}
    recommendations, _ = _evaluate(model, patient_model_df, feature_order, scaler, feature_name_map,
                                   intervention_threshold, min_absolute_reduction, modes, exclude_fasting_over_1_day)
    single = recommendations.loc[recommendations["Number of Changed Factors"].eq(1)].reset_index(drop=True)
    combined = recommendations.loc[recommendations["Number of Changed Factors"].gt(1)].reset_index(drop=True)
    return single, combined, recommendations


def add_display_columns(df):
    if df.empty: return df
    df = df.copy()
    df["Original Risk"] = df["Original Predicted Probability"].apply(format_probability)
    df["Post-Intervention Risk"] = df["Post-counterfactual Prediction Probability"].apply(format_probability)
    df["Absolute Risk Reduction (%)"] = (df["Absolute Risk Reduction"]*100).round(1)
    df["Relative Risk Reduction (%)"] = (df["Relative Risk Reduction"]*100).round(1)
    return df
