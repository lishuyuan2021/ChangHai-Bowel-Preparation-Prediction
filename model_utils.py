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
    used_features = set()
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
        sex_options = {"Male": 1, "Female": 0}
        selected_sex = st.selectbox("Sex", list(sex_options.keys()))

    with c3:
        bmi = st.number_input("BMI(kg/m²)", min_value=10.0, max_value=50.0, value=23.0, step=0.1)

    if "Age" in row:
        row["Age"] = standardize_value(scaler, "Age", age)
        used_features.add("Age")
    if "Sex" in row:
        row["Sex"] = sex_options[selected_sex]
        used_features.add("Sex")
    if "BMI" in row:
        row["BMI"] = standardize_value(scaler, "BMI", bmi)
        used_features.add("BMI")

    raw_values.update({"Age": age, "Sex": sex_options[selected_sex], "BMI": bmi})
    summary_rows.extend([
        {"Variable": "Age", "Value": age},
        {"Variable": "BMI", "Value": bmi},
        {"Variable": "Sex", "Value": selected_sex},
    ])

    st.subheader("Clinical factors")
    clinical_configs = {

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
        
        "StoolForm": {
            "label": "Usual stool form",
            "options": {"Bristol 3–7": 0, "Bristol 1–2 / hard": 1}
        },

        "DiabetesMellitus": {
            "label": "Diabetes mellitus",
            "options": {"No": 0, "Yes": 1}
        },

        "PreviousAbdominopelvicSurgery_1": {
            "label": "Previous abdominal surgery variable",
            "options": {"No": 0, "Yes": 1}
        },
    }

    clinical_cols = st.columns(3)
    shown_idx = 0

    for feature, cfg in clinical_configs.items():
        if feature not in feature_order:
            continue

        with clinical_cols[shown_idx % 3]:
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
        shown_idx += 1
    # ------------------------------------------------------------
    # 3. 肠道准备相关因素
    # ------------------------------------------------------------
    st.subheader("Bowel preparation details")

    bowel_cols1 = st.columns(3)
    bowel_cols2 = st.columns(4)

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
        with bowel_cols1[0]:
            selected_diet = st.selectbox("Dietary restriction strategy", list(valid_diet_options.keys()))
        row = set_onehot(row, dietary_group, valid_diet_options[selected_diet])
        summary_rows.append({"Variable": "Dietary restriction strategy", "Value": selected_diet})
        used_features.update(dietary_group)

    # 饮食限制天数
    with bowel_cols1[1]:
        diet_days = st.number_input("Dietary restriction days", min_value=0.0, max_value=7.0, value=1.0, step=1.0)

    if "DietaryRestrictionDays" in row:
        row["DietaryRestrictionDays"] = standardize_value(scaler, "DietaryRestrictionDays", diet_days)
        raw_values["DietaryRestrictionDays"] = diet_days
        summary_rows.append({"Variable": "Dietary restriction days", "Value": f"{diet_days} days"})
        used_features.add("DietaryRestrictionDays")

    # 泻药方案
    if laxative_group:
        valid_lax_options = {k: v for k, v in laxative_options.items() if v in laxative_group}
        with bowel_cols1[2]:
            selected_lax = st.selectbox("Laxative regimen", list(valid_lax_options.keys()))
        row = set_onehot(row, laxative_group, valid_lax_options[selected_lax])
        summary_rows.append({"Variable": "Laxative regimen", "Value": selected_lax})
        used_features.update(laxative_group)

    # 泻药是否分次服用
    if "SplitDose_BP" in feature_order:
        with bowel_cols2[0]:
            split_options = {"No": 0, "Yes": 1}
            selected_split = st.selectbox("Taking laxatives in divided doses", list(split_options.keys()))
        row["SplitDose_BP"] = split_options[selected_split]
        raw_values["SplitDose_BP"] = split_options[selected_split]
        summary_rows.append({"Variable": "Taking laxatives in divided doses", "Value": selected_split})
        used_features.add("SplitDose_BP")
    # 肠道准备宣教方式
    if "BPEducationModality" in feature_order:
        with bowel_cols2[1]:
            edu_options = {"Text + Illustrated or Video Instructions": 0, "Oral or Written Instructions": 1}
            selected_edu = st.selectbox("Educational Methods for Bowel Preparation", list(edu_options.keys()))
        row["BPEducationModality"] = edu_options[selected_edu]
        raw_values["BPEducationModality"] = edu_options[selected_edu]
        summary_rows.append({"Variable": "Educational Methods for Bowel Preparation", "Value": selected_edu})
        used_features.add("BPEducationModality")

    # 服用泻药后是否加强活动
    if "PreColonoscopyPhysicalActivity" in feature_order:
        with bowel_cols2[2]:
            activity_options = {"No": 0, "Yes": 1}
            selected_activity = st.selectbox("Physical Activity After Laxative Use", list(activity_options.keys()))
        row["PreColonoscopyPhysicalActivity"] = activity_options[selected_activity]
        raw_values["PreColonoscopyPhysicalActivity"] = activity_options[selected_activity]
        summary_rows.append({"Variable": "Physical Activity After Laxative Use", "Value": selected_activity})
        used_features.add("PreColonoscopyPhysicalActivity")

    # 肠道准备至肠镜检查时间间隔
    if interval_group:
        valid_interval_options = {k: v for k, v in interval_options.items() if v in interval_group}
        with bowel_cols2[3]:
            selected_interval = st.selectbox("Time Interval from Bowel Preparation to Colonoscopy", list(valid_interval_options.keys()))
        row = set_onehot(row, interval_group, valid_interval_options[selected_interval])
        summary_rows.append({"Variable": "Time Interval from Bowel Preparation to Colonoscopy", "Value": selected_interval})
        used_features.update(interval_group)

    # ------------------------------------------------------------
    # 未展示但模型需要的变量：默认 0
    # ------------------------------------------------------------
    for col in feature_order:
        if col not in used_features:
            row[col] = row.get(col, 0)

    patient_model_df = pd.DataFrame([row])[feature_order]
    patient_raw_df = pd.DataFrame([raw_values])
    input_summary_df = pd.DataFrame(summary_rows)

    return patient_raw_df, patient_model_df, input_summary_df

