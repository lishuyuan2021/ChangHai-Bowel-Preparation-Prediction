# ============================================================
# Counterfactual / Intervention Scenario Utilities
# Deterministic scan of modifiable features
# ============================================================

import itertools
import numpy as np
import pandas as pd

from model_utils import (
    format_probability,
    standardize_value,
    inverse_standardized_value,
)


DIET_DAYS_ORIGINAL_MIN = 0
DIET_DAYS_ORIGINAL_MAX = 3


dietary_label_map = {
    "DietaryRestriction_1": "Fasting",
    "DietaryRestriction_2": "Low-residue Diet",
    "DietaryRestriction_3": "Liquid Diet",
    "DietaryRestriction_4": "Normal Diet"
}

laxative_label_map = {
    "LaxativeRegimen_1": "PEG 2L",
    "LaxativeRegimen_2": "PEG 3L",
    "LaxativeRegimen_3": "PEG 4L",
    "LaxativeRegimen_4": "Sodium Phosphate",
    "LaxativeRegimen_5": "Mannitol",
    "LaxativeRegimen_6": "Magnesium Sulfate"
}

interval_label_map = {
    "BPtoColonoscopyinterval_1": "<120 min",
    "BPtoColonoscopyinterval_2": "120–240 min",
    "BPtoColonoscopyinterval_3": "240–360 min",
    "BPtoColonoscopyinterval_4": "≥360 min"
}

binary_label_map = {
    "BPEducationModality": {
        0: "Written + graphic/video education",
        1: "Oral or written education"
    },
    "SplitDose_BP": {
        0: "No split-dose bowel preparation",
        1: "Split-dose bowel preparation"
    },
    "PreColonoscopyPhysicalActivity": {
        0: "No pre-procedure physical activity",
        1: "Pre-procedure physical activity"
    }
}


def predict_single_risk(model, row, feature_order):
    row_df = pd.DataFrame([row])[feature_order]
    prob = model.predict_proba(row_df)[0, 1]
    return float(prob)


def set_onehot_group(row, group, target_col):
    row = row.copy()

    for col in group:
        if col in row.index:
            row[col] = 0

    if target_col in row.index:
        row[target_col] = 1

    return row


def decode_onehot_group(row, group, label_map):
    group = [col for col in group if col in row.index]

    if len(group) == 0:
        return None

    active_cols = [
        col for col in group
        if int(round(float(row[col]))) == 1
    ]

    if len(active_cols) == 0:
        return "None"

    return label_map.get(active_cols[0], active_cols[0])


def get_binary_label(feature, value):
    value = int(round(float(value)))

    if feature in binary_label_map:
        return binary_label_map[feature].get(value, value)

    return value


def set_scalar_value(row, col, value):
    row = row.copy()
    row[col] = value
    return row


def get_diet_days(row, scaler):
    if "DietaryRestrictionDays" not in row.index:
        return None

    return round(
        float(
            inverse_standardized_value(
                scaler,
                "DietaryRestrictionDays",
                row["DietaryRestrictionDays"]
            )
        ),
        1
    )


def set_diet_days(row, scaler, target_days):
    row = row.copy()

    if "DietaryRestrictionDays" in row.index:
        row["DietaryRestrictionDays"] = standardize_value(
            scaler,
            "DietaryRestrictionDays",
            target_days
        )

    return row


def classify_suggestion(prob, intervention_threshold):
    if prob < intervention_threshold:
        return "Risk decreased below intervention threshold"
    return "Risk decreased but remained above intervention threshold"



def postprocess_row(row, feature_order):
    row = row.copy()
    return row[feature_order]


def is_fasting_over_1_day(row, feature_order, scaler):
    """Determine whether the current recommendation is “fast for 1 day or more.”"""
    dietary_group = [c for c in [
        "DietaryRestriction_1",
        "DietaryRestriction_2",
        "DietaryRestriction_3",
        "DietaryRestriction_4"
    ] if c in feature_order]

    diet = decode_onehot_group(row, dietary_group, dietary_label_map)
    days = get_diet_days(row, scaler)

    return diet == "Fasting" and days is not None and days >= 1


def generate_single_intervention_candidates(
    original_row,
    feature_order,
    scaler,
    feature_name_map,
    exclude_fasting_over_1_day=True
):
    actions = []

    dietary_group = [c for c in [
        "DietaryRestriction_1",
        "DietaryRestriction_2",
        "DietaryRestriction_3",
        "DietaryRestriction_4"
    ] if c in feature_order]

    laxative_group = [c for c in [
        "LaxativeRegimen_1",
        "LaxativeRegimen_2",
        "LaxativeRegimen_3",
        "LaxativeRegimen_4",
        "LaxativeRegimen_5",
        "LaxativeRegimen_6"
    ] if c in feature_order]

    interval_group = [c for c in [
        "BPtoColonoscopyinterval_1",
        "BPtoColonoscopyinterval_2",
        "BPtoColonoscopyinterval_3",
        "BPtoColonoscopyinterval_4"
    ] if c in feature_order]

    # 饮食限制天数
    if "DietaryRestrictionDays" in feature_order:
        original_days = get_diet_days(original_row, scaler)

        for target_days in range(DIET_DAYS_ORIGINAL_MIN, DIET_DAYS_ORIGINAL_MAX + 1):
            if original_days is None:
                continue

            if abs(float(target_days) - float(original_days)) < 1e-6:
                continue

            direction = "increase" if target_days > original_days else "decrease"

            actions.append({
                "Intervention Type": "Dietary restriction duration",
                "Counterfactual Recommendation": f"{direction}dietary restriction duration to {target_days} days",
                "Baseline Measure": f"{original_days} days",
                "Post-Intervention Measure": f"{target_days} days",
                "Changed Features": "DietaryRestrictionDays",
                "apply": lambda row, d=target_days: set_diet_days(row, scaler, d)
            })

    # 二元可干预变量
    for col in [
        "BPEducationModality",
        "SplitDose_BP",
        "PreColonoscopyPhysicalActivity"
    ]:
        if col not in feature_order:
            continue

        old_value = int(round(float(original_row[col])))

        for new_value in [0, 1]:
            if new_value == old_value:
                continue

            old_label = get_binary_label(col, old_value)
            new_label = get_binary_label(col, new_value)

            actions.append({
                "Intervention Type": feature_name_map.get(col, col),
                "Counterfactual Recommendation": f"{feature_name_map.get(col, col)}:{old_label} → {new_label}",
                "Baseline Measure": old_label,
                "Post-Intervention Measure": new_label,
                "Changed Features": col,
                "apply": lambda row, c=col, v=new_value: set_scalar_value(row, c, v)
            })

    # 饮食策略
    if dietary_group:
        old_diet = decode_onehot_group(original_row, dietary_group, dietary_label_map)

        for target_col in dietary_group:
            target_label = dietary_label_map.get(target_col, target_col)

            if exclude_fasting_over_1_day and target_label == "Fasting" and get_diet_days(original_row, scaler) is not None and get_diet_days(original_row, scaler) >= 1:
                continue

            if target_label == old_diet:
                continue

            actions.append({
                "Intervention Type": "Dietary restriction strategy",
                "Counterfactual Recommendation": f"Dietary strategy: {old_diet} → {target_label}",
                "Baseline Measure": old_diet,
                "Post-Intervention Measure": target_label,
                "Changed Features": ",".join(dietary_group),
                "apply": lambda row, g=dietary_group, t=target_col: set_onehot_group(row, g, t)
            })

    # 泻药方案
    if laxative_group:
        old_laxative = decode_onehot_group(original_row, laxative_group, laxative_label_map)

        for target_col in laxative_group:
            target_label = laxative_label_map.get(target_col, target_col)

            if target_label == old_laxative:
                continue

            if old_laxative == "PEG 2L" and target_label in ["PEG 3L", "PEG 4L"]:
                category = "Laxative regimen / dose escalation"
                intervention_text = f"Increase laxative dose: {old_laxative} → {target_label}"
            elif old_laxative == "PEG 3L" and target_label == "PEG 4L":
                category = "Laxative regimen / dose escalation"
                intervention_text = f"Increase laxative dose: {old_laxative} → {target_label}"
            else:
                category = "Laxative regimen"
                intervention_text = f"Laxative regimen: {old_laxative} → {target_label}"

            actions.append({
                "Intervention Type": category,
                "Counterfactual Recommendation": intervention_text,
                "Baseline Measure": old_laxative,
                "Post-Intervention Measure": target_label,
                "Changed Features": ",".join(laxative_group),
                "apply": lambda row, g=laxative_group, t=target_col: set_onehot_group(row, g, t)
            })

    # 肠道准备至肠镜检查时间间隔
    if interval_group:
        old_interval = decode_onehot_group(original_row, interval_group, interval_label_map)

        for target_col in interval_group:
            target_label = interval_label_map.get(target_col, target_col)

            if target_label == old_interval:
                continue

            actions.append({
                "Intervention Type": "Interval between bowel preparation and colonoscopy",
                "Counterfactual Recommendation": f"Interval to colonoscopy: {old_interval} → {target_label}",
                "Baseline Measure": old_interval,
                "Post-Intervention Measure": target_label,
                "Changed Features": ",".join(interval_group),
                "apply": lambda row, g=interval_group, t=target_col: set_onehot_group(row, g, t)
            })

    return actions


def evaluate_single_interventions(
    model,
    patient_model_df,
    patient_raw_df,
    feature_order,
    scaler,
    feature_name_map,
    intervention_threshold,
    min_absolute_reduction=0.0,
    exclude_fasting_over_1_day=True
):
    original_row = patient_model_df.iloc[0][feature_order].copy()
    original_prob = predict_single_risk(model, original_row, feature_order)

    actions = generate_single_intervention_candidates(
        original_row=original_row,
        feature_order=feature_order,
        scaler=scaler,
        feature_name_map=feature_name_map,
        exclude_fasting_over_1_day=exclude_fasting_over_1_day
    )

    rows = []
    all_rows = []

    for i, action in enumerate(actions, start=1):
        cf_row = action["apply"](original_row)
        cf_row = postprocess_row(cf_row, feature_order)

        if exclude_fasting_over_1_day and "DietaryRestriction" in action["Changed Features"]:
            if is_fasting_over_1_day(cf_row, feature_order, scaler):
                continue

        cf_prob = predict_single_risk(model, cf_row, feature_order)

        abs_red = original_prob - cf_prob
        rel_red = abs_red / original_prob if original_prob > 0 else np.nan
        risk_decreasing = abs_red > min_absolute_reduction

        scenario_row = {
            "ID": i,
            "Category": "Single intervention",
            "Intervention Type": action["Intervention Type"],
            "Counterfactual Recommendation": action["Counterfactual Recommendation"],
            "Changed Features": action["Changed Features"],
            "Baseline Measure": action["Baseline Measure"],
            "Post-Intervention Measure": action["Post-Intervention Measure"],
            "Original Predicted Probability": original_prob,
            "Post-counterfactual Prediction Probability": cf_prob,
            "Absolute Risk Reduction": abs_red,
            "Relative Risk Reduction": rel_red,
            "Risk Decreasing": risk_decreasing,
            "Below Intervention Threshold": cf_prob < intervention_threshold,
            "Interpretation": classify_suggestion(cf_prob, intervention_threshold) if risk_decreasing else "Risk did not decrease"
        }

        all_rows.append(scenario_row)

        if risk_decreasing:
            rows.append(scenario_row)

    df = pd.DataFrame(rows)
    all_df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.sort_values(
            by="Absolute Risk Reduction",
            ascending=False
        ).reset_index(drop=True)

    return df, all_df


def evaluate_pairwise_interventions(
    model,
    patient_model_df,
    patient_raw_df,
    feature_order,
    scaler,
    feature_name_map,
    intervention_threshold,
    min_absolute_reduction=0.0,
    exclude_fasting_over_1_day=True
):
    original_row = patient_model_df.iloc[0][feature_order].copy()
    original_prob = predict_single_risk(model, original_row, feature_order)

    actions = generate_single_intervention_candidates(
        original_row=original_row,
        feature_order=feature_order,
        scaler=scaler,
        feature_name_map=feature_name_map,
        exclude_fasting_over_1_day=exclude_fasting_over_1_day
    )

    rows = []
    all_rows = []

    for pair_id, (action_a, action_b) in enumerate(itertools.combinations(actions, 2), start=1):
        cf_row = original_row.copy()

        cf_row = action_a["apply"](cf_row)
        cf_row = action_b["apply"](cf_row)

        cf_row = postprocess_row(cf_row, feature_order)

        changed_features = action_a["Changed Features"] + " + " + action_b["Changed Features"]
        if exclude_fasting_over_1_day and "DietaryRestriction" in changed_features:
            if is_fasting_over_1_day(cf_row, feature_order, scaler):
                continue

        cf_prob = predict_single_risk(model, cf_row, feature_order)

        abs_red = original_prob - cf_prob
        rel_red = abs_red / original_prob if original_prob > 0 else np.nan
        risk_decreasing = abs_red > min_absolute_reduction

        scenario_row = {
            "ID": pair_id,
            "Category": "Pairwise intervention",
            "Intervention Type": "Combined intervention",
            "Counterfactual Recommendation": action_a["Counterfactual Recommendation"] + " + " + action_b["Counterfactual Recommendation"],
            "Changed Features": action_a["Changed Features"] + " + " + action_b["Changed Features"],
            "Baseline Measure": str(action_a["Baseline Measure"]) + " + " + str(action_b["Baseline Measure"]),
            "Post-Intervention Measure": str(action_a["Post-Intervention Measure"]) + " + " + str(action_b["Post-Intervention Measure"]),
            "Original Predicted Probability": original_prob,
            "Post-counterfactual Prediction Probability": cf_prob,
            "Absolute Risk Reduction": abs_red,
            "Relative Risk Reduction": rel_red,
            "Risk Decreasing": risk_decreasing,
            "Below Intervention Threshold": cf_prob < intervention_threshold,
            "Interpretation": classify_suggestion(cf_prob, intervention_threshold) if risk_decreasing else "Risk did not decrease"
        }

        all_rows.append(scenario_row)

        if risk_decreasing:
            rows.append(scenario_row)

    df = pd.DataFrame(rows)
    all_df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.sort_values(
            by="Absolute Risk Reduction",
            ascending=False
        ).reset_index(drop=True)

    return df, all_df


def scan_risk_decreasing_measures(
    model,
    patient_model_df,
    patient_raw_df,
    feature_order,
    scaler,
    feature_name_map,
    intervention_threshold=0.135,
    min_absolute_reduction=0.0,
    evaluate_pairwise=True,
    exclude_fasting_over_1_day=True
):
    single_df, _ = evaluate_single_interventions(
        model=model,
        patient_model_df=patient_model_df,
        patient_raw_df=patient_raw_df,
        feature_order=feature_order,
        scaler=scaler,
        feature_name_map=feature_name_map,
        intervention_threshold=intervention_threshold,
        min_absolute_reduction=min_absolute_reduction,
        exclude_fasting_over_1_day=exclude_fasting_over_1_day
    )

    if evaluate_pairwise:
        pairwise_df, _ = evaluate_pairwise_interventions(
            model=model,
            patient_model_df=patient_model_df,
            patient_raw_df=patient_raw_df,
            feature_order=feature_order,
            scaler=scaler,
            feature_name_map=feature_name_map,
            intervention_threshold=intervention_threshold,
            min_absolute_reduction=min_absolute_reduction,
            exclude_fasting_over_1_day=exclude_fasting_over_1_day
        )
    else:
        pairwise_df = pd.DataFrame()

    all_list = []

    if not single_df.empty:
        all_list.append(single_df)

    if not pairwise_df.empty:
        all_list.append(pairwise_df)

    if all_list:
        all_df = pd.concat(all_list, axis=0, ignore_index=True)
        all_df = all_df.sort_values(
            by="Absolute Risk Reduction",
            ascending=False
        ).reset_index(drop=True)
    else:
        all_df = pd.DataFrame()

    return single_df, pairwise_df, all_df


def add_display_columns(df):
    if df.empty:
        return df

    df = df.copy()

    df["Original Risk"] = df["Original Predicted Probability"].apply(format_probability)
    df["Post-Intervention Risk"] = df["Post-counterfactual Prediction Probability"].apply(format_probability)

    df["Absolute Risk Reduction (%)"] = (
        df["Absolute Risk Reduction"] * 100
    ).round(1)

    df["Relative Risk Reduction (%)"] = (
        df["Relative Risk Reduction"] * 100
    ).round(1)

    return df
