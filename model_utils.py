# ============================================================
# Model and Input Utilities
# ============================================================

from pathlib import Path
import os
import cloudpickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Load deployment package
# ============================================================

@st.cache_resource
def load_deploy_pack(deploy_dir: str):
    deploy_dir = Path(deploy_dir)

    pack_path = deploy_dir / "Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl"

    if not pack_path.exists():
        raise FileNotFoundError(f"Deployment package not found: {pack_path}")

    with open(pack_path, "rb") as f:
        deploy_pack = cloudpickle.load(f)

    if deploy_pack.get("scaler", None) is None:
        scaler_path = deploy_dir / "standard_scaler_v2.pkl"
        if scaler_path.exists():
            deploy_pack["scaler"] = joblib.load(scaler_path)

    return deploy_pack


# ============================================================
# Display names
# ============================================================

def get_feature_display_map():
    return {
        "Age": "Age",
        "BMI": "BMI",
        "DietaryRestrictionDays": "Dietary Restriction Duration (Days)",

        "HospitalGrade": "Hospital Level",
        "Sex": "Sex",
        "InpatientStatus": "Inpatient Status",
        "PreviousColonoscopy": "Previous Colonoscopy",
        "ChronicConstipation": "Chronic Constipation",
        "ChronicDiarrhea": "Chronic Diarrhea",
        "DiabetesMellitus": "Diabetes Mellitus",
        "StoolForm": "Hard Stool",
        "BPEducationModality": "Bowel Preparation Education Modality",
        "SplitDose_BP": "Split-dose Bowel Preparation",
        "PreColonoscopyPhysicalActivity": "Pre-procedure Physical Activity",

        "BPtoColonoscopyinterval_1": "Interval to Colonoscopy: <120 min",
        "BPtoColonoscopyinterval_2": "Interval to Colonoscopy: 120–240 min",
        "BPtoColonoscopyinterval_3": "Interval to Colonoscopy: 240–360 min",
        "BPtoColonoscopyinterval_4": "Interval to Colonoscopy: ≥360 min",

        "DietaryRestriction_1": "Dietary Restriction: Fasting",
        "DietaryRestriction_2": "Dietary Restriction: Low-residue Diet",
        "DietaryRestriction_3": "Dietary Restriction: Liquid Diet",
        "DietaryRestriction_4": "Dietary Restriction: Normal Diet",

        "LaxativeRegimen_1": "Laxative Regimen: PEG 2L",
        "LaxativeRegimen_2": "Laxative Regimen: PEG 3L",
        "LaxativeRegimen_3": "Laxative Regimen: PEG 4L",
        "LaxativeRegimen_4": "Laxative Regimen: Sodium Phosphate",
        "LaxativeRegimen_5": "Laxative Regimen: Mannitol",
        "LaxativeRegimen_6": "Laxative Regimen: Magnesium Sulfate",

        "PsychotropicMedication_2": "Psychotropic Medication: TCA",
        "PreviousAbdominopelvicSurgery_1": "Previous Abdominopelvic Surgery"
    }


# ============================================================
# Continuous variable scaling
# ============================================================

def get_scaler_feature_order(scaler):
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return ["Age", "BMI", "DietaryRestrictionDays"]


def standardize_value(scaler, feature_name, original_value):
    scaler_feature_order = get_scaler_feature_order(scaler)

    if feature_name not in scaler_feature_order:
        return original_value

    idx = scaler_feature_order.index(feature_name)

    return (float(original_value) - scaler.mean_[idx]) / scaler.scale_[idx]


def inverse_standardized_value(scaler, feature_name, standardized_value):
    scaler_feature_order = get_scaler_feature_order(scaler)

    if feature_name not in scaler_feature_order:
        return standardized_value

    idx = scaler_feature_order.index(feature_name)

    return float(standardized_value) * scaler.scale_[idx] + scaler.mean_[idx]


# ============================================================
# Helpers
# ============================================================

def set_onehot(row, group, selected_col):
    for col in group:
        if col in row:
            row[col] = 0
    if selected_col in row:
        row[selected_col] = 1
    return row


def format_probability(prob):
    if pd.isna(prob):
        return "Not applicable"
    return f"{float(prob) * 100:.1f}%"


def classify_binary_risk(prob, threshold=0.135):
    return "High Risk" if prob >= threshold else "Low Risk"


def predict_patient(model, patient_model_df, feature_order):
    patient_model_df = patient_model_df[feature_order]
    return float(model.predict_proba(patient_model_df)[:, 1][0])


# ============================================================
# Streamlit form
# ============================================================

def build_patient_input_form(feature_order, scaler, feature_name_map):
    """
    Build clinical input form and return:
    1. patient_raw_df: raw clinical values for display
    2. patient_model_df: encoded/scaled model input
    3. input_summary_df: readable input summary
    """

    row = {col: 0 for col in feature_order}
    raw_values = {}
    summary_rows = []

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

    st.subheader("Basic information")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age, years", min_value=18.0, max_value=100.0, value=60.0, step=1.0)

    with c2:
        bmi = st.number_input("BMI, kg/m²", min_value=10.0, max_value=50.0, value=23.0, step=0.1)

    with c3:
        diet_days = st.number_input("Dietary restriction duration, days", min_value=0.0, max_value=7.0, value=1.0, step=1.0)

    if "Age" in row:
        row["Age"] = standardize_value(scaler, "Age", age)
    if "BMI" in row:
        row["BMI"] = standardize_value(scaler, "BMI", bmi)
    if "DietaryRestrictionDays" in row:
        row["DietaryRestrictionDays"] = standardize_value(scaler, "DietaryRestrictionDays", diet_days)

    raw_values.update({
        "Age": age,
        "BMI": bmi,
        "DietaryRestrictionDays": diet_days
    })

    summary_rows.extend([
        {"Variable": "Age", "Value": age},
        {"Variable": "BMI", "Value": bmi},
        {"Variable": "Dietary restriction duration", "Value": f"{diet_days} days"},
    ])

    st.subheader("Clinical factors")

    col1, col2, col3 = st.columns(3)

    binary_configs = {
        "HospitalGrade": {
            "label": "Hospital level",
            "options": {"Non-tertiary / lower-level hospital": 0, "Tertiary hospital": 1}
        },
        "Sex": {
            "label": "Sex coding used in training",
            "options": {"0": 0, "1": 1}
        },
        "InpatientStatus": {
            "label": "Patient setting",
            "options": {"Outpatient": 0, "Inpatient": 1}
        },
        "PreviousColonoscopy": {
            "label": "Previous colonoscopy",
            "options": {"No": 0, "Yes": 1}
        },
        "ChronicConstipation": {
            "label": "Chronic constipation",
            "options": {"No": 0, "Yes": 1}
        },
        "ChronicDiarrhea": {
            "label": "Chronic diarrhea",
            "options": {"No": 0, "Yes": 1}
        },
        "DiabetesMellitus": {
            "label": "Diabetes mellitus",
            "options": {"No": 0, "Yes": 1}
        },
        "StoolForm": {
            "label": "Usual stool form",
            "options": {"Bristol 3–7": 0, "Bristol 1–2 / hard stool": 1}
        },
        "BPEducationModality": {
            "label": "Bowel preparation education modality",
            "options": {"Written + graphic/video education": 0, "Oral or written education": 1}
        },
        "SplitDose_BP": {
            "label": "Split-dose bowel preparation",
            "options": {"No": 0, "Yes": 1}
        },
        "PreColonoscopyPhysicalActivity": {
            "label": "Physical activity before colonoscopy",
            "options": {"No": 0, "Yes": 1}
        },
        "PsychotropicMedication_2": {
            "label": "Psychotropic medication: TCA",
            "options": {"No": 0, "Yes": 1}
        },
        "PreviousAbdominopelvicSurgery_1": {
            "label": "Previous abdominopelvic surgery variable",
            "options": {"0": 0, "1": 1}
        },
    }

    columns_cycle = [col1, col2, col3]
    i = 0

    used_features = set(["Age", "BMI", "DietaryRestrictionDays"])

    for feature, cfg in binary_configs.items():
        if feature not in feature_order:
            continue

        with columns_cycle[i % 3]:
            selected = st.selectbox(
                cfg["label"],
                options=list(cfg["options"].keys()),
                index=0,
                key=f"input_{feature}"
            )

        value = cfg["options"][selected]
        row[feature] = value
        raw_values[feature] = value
        summary_rows.append({"Variable": cfg["label"], "Value": selected})
        used_features.add(feature)
        i += 1

    st.subheader("Bowel preparation details")

    dcol, lcol, icol = st.columns(3)

    dietary_options = {
        "Fasting": "DietaryRestriction_1",
        "Low-residue diet": "DietaryRestriction_2",
        "Liquid diet": "DietaryRestriction_3",
        "Normal diet": "DietaryRestriction_4",
    }

    laxative_options = {
        "PEG 2L": "LaxativeRegimen_1",
        "PEG 3L": "LaxativeRegimen_2",
        "PEG 4L": "LaxativeRegimen_3",
        "Sodium phosphate": "LaxativeRegimen_4",
        "Mannitol": "LaxativeRegimen_5",
        "Magnesium sulfate": "LaxativeRegimen_6",
    }

    interval_options = {
        "<120 min": "BPtoColonoscopyinterval_1",
        "120–240 min": "BPtoColonoscopyinterval_2",
        "240–360 min": "BPtoColonoscopyinterval_3",
        "≥360 min": "BPtoColonoscopyinterval_4",
    }

    if dietary_group:
        valid_diet_options = {k: v for k, v in dietary_options.items() if v in dietary_group}
        with dcol:
            selected_diet = st.selectbox("Dietary restriction strategy", list(valid_diet_options.keys()))
        row = set_onehot(row, dietary_group, valid_diet_options[selected_diet])
        summary_rows.append({"Variable": "Dietary restriction strategy", "Value": selected_diet})
        used_features.update(dietary_group)

    if laxative_group:
        valid_lax_options = {k: v for k, v in laxative_options.items() if v in laxative_group}
        with lcol:
            selected_lax = st.selectbox("Laxative regimen", list(valid_lax_options.keys()))
        row = set_onehot(row, laxative_group, valid_lax_options[selected_lax])
        summary_rows.append({"Variable": "Laxative regimen", "Value": selected_lax})
        used_features.update(laxative_group)

    if interval_group:
        valid_interval_options = {k: v for k, v in interval_options.items() if v in interval_group}
        with icol:
            selected_interval = st.selectbox("Interval from bowel preparation to colonoscopy", list(valid_interval_options.keys()))
        row = set_onehot(row, interval_group, valid_interval_options[selected_interval])
        summary_rows.append({"Variable": "Interval to colonoscopy", "Value": selected_interval})
        used_features.update(interval_group)

    remaining_binary = [
        col for col in feature_order
        if col not in used_features
    ]

    if remaining_binary:
        with st.expander("Additional model variables", expanded=False):
            for col in remaining_binary:
                value = st.selectbox(
                    feature_name_map.get(col, col),
                    options=[0, 1],
                    index=0,
                    key=f"additional_{col}"
                )
                row[col] = value
                summary_rows.append({"Variable": feature_name_map.get(col, col), "Value": value})

    patient_model_df = pd.DataFrame([row])[feature_order]
    patient_raw_df = pd.DataFrame([raw_values])
    input_summary_df = pd.DataFrame(summary_rows)

    return patient_raw_df, patient_model_df, input_summary_df
