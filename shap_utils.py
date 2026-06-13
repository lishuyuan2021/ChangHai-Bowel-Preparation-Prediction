# ============================================================
# SHAP Utilities
# 修正版：避免 Streamlit cache_resource 对 model / DataFrame hash 报错
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st


def make_predict_function(model, feature_order):
    def predict_poor_prep_probability(X_input):
        if isinstance(X_input, pd.DataFrame):
            X_df = X_input.copy()
        else:
            X_df = pd.DataFrame(
                X_input,
                columns=feature_order
            )

        X_df = X_df[feature_order]

        return model.predict_proba(X_df)[:, 1]

    return predict_poor_prep_probability


# ============================================================
# 关键修正：
# _model 和 _background_data 前面加下划线
# Streamlit 会忽略以下划线开头的参数，不再尝试 hash 它们
# ============================================================

@st.cache_resource(show_spinner=False)
def get_shap_explainer(
    _model,
    _background_data,
    feature_order,
    background_n=80,
    random_state=2026
):
    background_data = _background_data[feature_order].copy()

    if len(background_data) > background_n:
        background_data = shap.sample(
            background_data,
            background_n,
            random_state=random_state
        )

    masker = shap.maskers.Independent(
        background_data
    )

    explainer = shap.Explainer(
        make_predict_function(_model, feature_order),
        masker,
        algorithm="permutation",
        seed=random_state
    )

    return explainer


def get_display_names(feature_order, feature_name_map):
    return [
        feature_name_map.get(col, col)
        for col in feature_order
    ]


def build_display_data(X_model_df, X_raw_df, feature_order):
    """
    SHAP values 使用模型输入计算。
    为了避免错位，这里展示值仍使用模型输入矩阵。
    """
    return X_model_df[feature_order].values


def compute_patient_shap(
    explainer,
    patient_model_df,
    patient_raw_df,
    feature_order,
    feature_name_map,
):
    patient_model_df = patient_model_df[feature_order].copy()

    max_evals = 2 * len(feature_order) + 1

    shap_values_raw = explainer(
        patient_model_df,
        max_evals=max_evals
    )

    feature_display_names = get_display_names(
        feature_order,
        feature_name_map
    )

    patient_shap_values = shap.Explanation(
        values=shap_values_raw.values,
        base_values=shap_values_raw.base_values,
        data=build_display_data(
            patient_model_df,
            patient_raw_df,
            feature_order
        ),
        feature_names=feature_display_names
    )

    return patient_shap_values


def compute_global_shap(
    explainer,
    X_global_source,
    feature_order,
    feature_name_map,
    sample_n=60,
    random_state=2026
):
    X_global_source = X_global_source[feature_order].copy()

    if len(X_global_source) > sample_n:
        X_global = shap.sample(
            X_global_source,
            sample_n,
            random_state=random_state
        )
    else:
        X_global = X_global_source.copy()

    max_evals = 2 * len(feature_order) + 1

    shap_values_raw = explainer(
        X_global,
        max_evals=max_evals
    )

    feature_display_names = get_display_names(
        feature_order,
        feature_name_map
    )

    global_shap_values = shap.Explanation(
        values=shap_values_raw.values,
        base_values=shap_values_raw.base_values,
        data=X_global[feature_order].values,
        feature_names=feature_display_names
    )

    return global_shap_values


def plot_patient_waterfall(
    patient_shap_values,
    predicted_prob,
    max_display=15
):
    plt.figure(figsize=(11, 7))

    shap.plots.waterfall(
        patient_shap_values[0],
        max_display=max_display,
        show=False
    )

    plt.title(
        f"Individual SHAP Waterfall Plot\nPredicted Risk = {predicted_prob:.3f}",
        fontsize=14,
        fontweight="bold",
        pad=18
    )

    plt.tight_layout()

    fig = plt.gcf()

    return fig


def plot_global_beeswarm(
    global_shap_values,
    max_display=20
):
    plt.figure(figsize=(11, 8))

    shap.plots.beeswarm(
        global_shap_values,
        max_display=max_display,
        show=False
    )

    plt.title(
        "SHAP Beeswarm Plot",
        fontsize=14,
        fontweight="bold",
        pad=18
    )

    plt.xlabel(
        "SHAP Value (Impact on Probability of Poor Bowel Preparation)",
        fontsize=11
    )

    plt.tight_layout()

    fig = plt.gcf()

    return fig