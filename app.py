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
    page_title="ChangHai Bowel Preparation Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 ChangHai Bowel Preparation Risk Prediction Tool")
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

DEPLOY_DIR = "Final_Deploy_Stacking_V2"
INTERVENTION_THRESHOLD = 0.135

# 默认显示 SHAP 蜂巢图，抽样样本量固定为 100
RUN_GLOBAL_SHAP = True
GLOBAL_SHAP_N = 100

# 默认进行两两组合干预扫描
RUN_PAIRWISE_CF = True

# 默认排除“禁食 1 天及以上”作为反事实建议目标
EXCLUDE_FASTING_OVER_1_DAY = True

# 只要预测风险下降即保留
MIN_ABSOLUTE_REDUCTION = 0.000

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


# ============================================================
# Input form
# ============================================================

st.header("1. Patient Variables")

patient_raw_df, patient_model_df, input_summary_df = build_patient_input_form(
    feature_order=feature_order,
    scaler=scaler,
    feature_name_map=feature_name_map,
)



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
            max_display=20
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
            exclude_fasting_over_1_day=EXCLUDE_FASTING_OVER_1_DAY
        )

    if cf_all_df.empty:
        st.info(
            "No model-based risk-decreasing intervention scenario was identified "
            "under the predefined modifiable feature constraints."
        )
    else:
        cf_all_display = add_display_columns(cf_all_df)

        st.subheader("Top 20 risk-decreasing suggestions")


        top_display_cols = [
            "Intervention Type",
            "Counterfactual Recommendation",
            "Original Risk",
            "Post-Intervention Risk"
        ]
        st.dataframe(
            cf_all_display[top_display_cols].head(20),
            use_container_width=True
        )

        detail_display_cols = [
            "Intervention Type",
            "Counterfactual Recommendation",
            "Baseline Measure",
            "Post-Intervention Measure",
            "Original Risk",
            "Post-Intervention Risk"
        ]

        with st.expander("Recommendations for Individual Interventions", expanded=True):
            if cf_single_df.empty:
                st.info("No individual interventions were found that could reduce the predicted risk.")
            else:
                st.dataframe(
                    add_display_columns(cf_single_df)[detail_display_cols],
                    use_container_width=True
                )

        with st.expander("Pairwise Intervention Suggestions", expanded=True):
            if cf_pairwise_df.empty:
                st.info("No pairwise interventions were found that could reduce the predicted risk, or pairwise scanning was disabled.")
            else:
                st.dataframe(
                    add_display_columns(cf_pairwise_df)[detail_display_cols].head(50),
                    use_container_width=True
                )

else:
    st.info("Enter patient variables and click 'Run Prediction'.")