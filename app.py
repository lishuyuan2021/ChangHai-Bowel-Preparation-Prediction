# ============================================================
# Streamlit App
# Poor Bowel Preparation Risk Prediction
# Final Model: Raw Stacking Classifier
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_utils import (
    load_deploy_pack,
    build_patient_input_form,
    predict_patient,
    classify_binary_risk,
    format_probability,
    get_feature_display_map,
)

from shap_utils import (
    get_shap_explainer,
    compute_patient_shap,
    compute_global_shap,
    plot_patient_waterfall,
    plot_global_beeswarm,
)

from counterfactual_utils import (
    scan_risk_decreasing_measures,
    add_display_columns,
)


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Poor Bowel Preparation Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Poor Bowel Preparation Risk Prediction Tool")
st.caption(
    "Final model: Raw Stacking Classifier. "
    "This tool estimates the risk of poor bowel preparation and provides model-based explanations."
)

st.warning(
    "This web tool is intended for research and decision-support only. "
    "It should not replace clinician judgement or local bowel-preparation protocols."
)


# ============================================================
# Sidebar settings
# ============================================================

st.sidebar.header("Model Settings")

DEPLOY_DIR = st.sidebar.text_input(
    "Deployment folder",
    value="Final_Deploy_Stacking_V2"
)

INTERVENTION_THRESHOLD = st.sidebar.number_input(
    "High-risk threshold",
    min_value=0.000,
    max_value=1.000,
    value=0.135,
    step=0.001,
    format="%.3f"
)

RUN_GLOBAL_SHAP = st.sidebar.checkbox(
    "Show SHAP beeswarm plot",
    value=True
)

GLOBAL_SHAP_N = st.sidebar.slider(
    "SHAP beeswarm sample size",
    min_value=20,
    max_value=200,
    value=60,
    step=10
)

RUN_PAIRWISE_CF = st.sidebar.checkbox(
    "Evaluate pairwise counterfactual suggestions",
    value=True
)

EXCLUDE_FASTING = st.sidebar.checkbox(
    "Exclude fasting from suggestions",
    value=False
)

MIN_ABSOLUTE_REDUCTION = st.sidebar.number_input(
    "Minimum absolute risk reduction for suggestions",
    min_value=0.000,
    max_value=0.200,
    value=0.000,
    step=0.001,
    format="%.3f"
)


# ============================================================
# Load model
# ============================================================

try:
    deploy_pack = load_deploy_pack(DEPLOY_DIR)
except Exception as e:
    st.error(f"Failed to load deployment package: {e}")
    st.stop()

model = deploy_pack["final_deploy_model"]
feature_order = deploy_pack["feature_order"]
deploy_info = deploy_pack["deploy_info"]
datasets = deploy_pack["datasets"]
scaler = deploy_pack.get("scaler", None)

if scaler is None:
    st.error(
        "Scaler was not found in the deployment package. "
        "The app needs the scaler to standardize Age, BMI, and DietaryRestrictionDays."
    )
    st.stop()

X_train_final = datasets["X_train_final"][feature_order]
X_val_final = datasets["X_val_final"][feature_order]

feature_name_map = get_feature_display_map()

st.sidebar.success("Model package loaded successfully.")


# ============================================================
# Input form
# ============================================================

st.header("1. Patient Variables")

patient_raw_df, patient_model_df, input_summary_df = build_patient_input_form(
    feature_order=feature_order,
    scaler=scaler,
    feature_name_map=feature_name_map,
)

with st.expander("View encoded model input", expanded=False):
    st.dataframe(patient_model_df.T.rename(columns={0: "Value"}), use_container_width=True)

with st.expander("View clinical input summary", expanded=False):
    st.dataframe(input_summary_df, use_container_width=True)


# ============================================================
# Prediction
# ============================================================

st.header("2. Prediction Result")

if st.button("Run Prediction", type="primary"):

    predicted_prob = predict_patient(
        model=model,
        patient_model_df=patient_model_df,
        feature_order=feature_order
    )

    risk_label = classify_binary_risk(
        predicted_prob,
        threshold=INTERVENTION_THRESHOLD
    )

    st.session_state["predicted_prob"] = predicted_prob
    st.session_state["risk_label"] = risk_label
    st.session_state["patient_model_df"] = patient_model_df
    st.session_state["patient_raw_df"] = patient_raw_df
    st.session_state["input_summary_df"] = input_summary_df

if "predicted_prob" in st.session_state:

    predicted_prob = st.session_state["predicted_prob"]
    risk_label = st.session_state["risk_label"]
    patient_model_df = st.session_state["patient_model_df"]
    patient_raw_df = st.session_state["patient_raw_df"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted probability of poor bowel preparation",
            format_probability(predicted_prob)
        )

    with col2:
        st.metric(
            "Risk category",
            risk_label
        )

    with col3:
        st.metric(
            "Intervention threshold",
            f"{INTERVENTION_THRESHOLD:.3f}"
        )

    if predicted_prob >= INTERVENTION_THRESHOLD:
        st.error("High risk: intensified bowel preparation may be considered.")
    else:
        st.success("Low risk: standard bowel preparation may be considered.")


    # ========================================================
    # SHAP explanation
    # ========================================================

    st.header("3. SHAP Explanation")

    with st.spinner("Preparing SHAP explainer..."):
        explainer = get_shap_explainer(
    _model=model,
    _background_data=X_train_final,
    feature_order=feature_order,
    background_n=80,
    random_state=2026
)

    tab1, tab2 = st.tabs(["Patient waterfall plot", "Global beeswarm plot"])

    with tab1:
        st.subheader("Individual SHAP Waterfall Plot")

        with st.spinner("Computing patient-level SHAP values..."):
            patient_shap_values = compute_patient_shap(
                explainer=explainer,
                patient_model_df=patient_model_df,
                patient_raw_df=patient_raw_df,
                feature_order=feature_order,
                feature_name_map=feature_name_map,
            )

        fig_waterfall = plot_patient_waterfall(
            patient_shap_values,
            predicted_prob=predicted_prob,
            max_display=15
        )

        st.pyplot(fig_waterfall, clear_figure=True)

    with tab2:
        st.subheader("SHAP Beeswarm Plot")

        if RUN_GLOBAL_SHAP:
            with st.spinner("Computing global SHAP values. This may take a while on first run..."):
                global_shap_values = compute_global_shap(
                    explainer=explainer,
                    X_global_source=X_val_final,
                    feature_order=feature_order,
                    feature_name_map=feature_name_map,
                    sample_n=GLOBAL_SHAP_N,
                    random_state=2026
                )

            fig_beeswarm = plot_global_beeswarm(
                global_shap_values,
                max_display=20
            )

            st.pyplot(fig_beeswarm, clear_figure=True)
        else:
            st.info("Enable 'Show SHAP beeswarm plot' in the sidebar to compute this plot.")


    # ========================================================
    # Counterfactual / intervention suggestions
    # ========================================================

    st.header("4. Counterfactual Intervention Suggestions")

    with st.spinner("Scanning risk-decreasing intervention scenarios..."):

        cf_single_df, cf_pairwise_df, cf_all_df = scan_risk_decreasing_measures(
            model=model,
            patient_model_df=patient_model_df,
            patient_raw_df=patient_raw_df,
            feature_order=feature_order,
            scaler=scaler,
            feature_name_map=feature_name_map,
            intervention_threshold=INTERVENTION_THRESHOLD,
            min_absolute_reduction=MIN_ABSOLUTE_REDUCTION,
            evaluate_pairwise=RUN_PAIRWISE_CF,
            exclude_fasting=EXCLUDE_FASTING
        )

    if cf_all_df.empty:
        st.info(
            "No model-based risk-decreasing intervention scenario was identified "
            "under the predefined modifiable feature constraints."
        )
    else:
        cf_all_display = add_display_columns(cf_all_df)

        st.subheader("Top risk-decreasing suggestions")

        display_cols = [
            "Scenario_Type",
            "Intervention",
            "Original_Risk",
            "Counterfactual_Risk",
            "Absolute_Risk_Reduction_pp",
            "Relative_Risk_Reduction_percent",
            "Below_Intervention_Threshold",
            "Interpretation"
        ]

        st.dataframe(
            cf_all_display[display_cols].head(20),
            use_container_width=True
        )

        csv_bytes = cf_all_display.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        st.download_button(
            label="Download counterfactual suggestions as CSV",
            data=csv_bytes,
            file_name="counterfactual_suggestions.csv",
            mime="text/csv"
        )

        with st.expander("Single intervention suggestions", expanded=False):
            if cf_single_df.empty:
                st.info("No single intervention decreased the predicted risk.")
            else:
                st.dataframe(
                    add_display_columns(cf_single_df),
                    use_container_width=True
                )

        with st.expander("Pairwise intervention suggestions", expanded=False):
            if cf_pairwise_df.empty:
                st.info("No pairwise intervention decreased the predicted risk, or pairwise scanning was disabled.")
            else:
                st.dataframe(
                    add_display_columns(cf_pairwise_df).head(50),
                    use_container_width=True
                )

else:
    st.info("Enter patient variables and click 'Run Prediction'.")
