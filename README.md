# Poor Bowel Preparation Risk Prediction Streamlit App

This repository deploys a Raw Stacking Classifier for predicting poor bowel preparation risk.

## Required model artifacts

Place the following folder in the project root:

```text
Final_Deploy_Stacking_V2/
├── Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl
├── standard_scaler_v2.pkl
├── deploy_info_v2.json
└── feature_order_v2.csv
```

The most important file is:

```text
Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl
```

## Project structure

```text
.
├── app.py
├── model_utils.py
├── shap_utils.py
├── counterfactual_utils.py
├── requirements.txt
├── README.md
└── Final_Deploy_Stacking_V2/
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Main outputs

The app provides:

1. Binary risk category: Low Risk or High Risk
2. Predicted probability of poor bowel preparation
3. SHAP waterfall plot for the current patient
4. SHAP beeswarm plot based on validation/background samples
5. Counterfactual-style intervention suggestions based on risk-decreasing scenario scans

## Notes

This application is intended for research and decision-support use only.
It should not replace clinician judgement or local clinical protocols.
