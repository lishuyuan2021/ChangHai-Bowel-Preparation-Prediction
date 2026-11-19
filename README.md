# Poor Bowel Preparation Risk Prediction Streamlit App

This app uses the original Raw Stacking Classifier and its 27 encoded/scaled features.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep the original `Final_Deploy_Stacking_V2/` directory in the repository root,
including `Final_Raw_Stacking_Deploy_Pack_V2_cloudpickle.pkl`,
`standard_scaler_v2.pkl`, `deploy_info_v2.json`, and `feature_order_v2.csv`.
The new `clinical_rules.py` must be present beside `app.py`.

## Patient inputs

- Choose **Enter BMI directly**, or **Calculate from height and weight**.
- In automatic mode, enter height in cm and weight in kg. BMI is calculated as
  `weight_kg / (height_cm / 100) ** 2` and updates when either input changes.
- Direct BMI entry does not require height or weight. Height and weight are
  used only to calculate BMI; they are not added to the model's 27 features.
- Both BMI methods retain the original accepted BMI range of 10–50.
- Dietary restriction duration is limited to 0–2 days. The step buttons and
  counterfactual search use a 0.5-day grid.
- **Normal diet** automatically sets duration to 0 and disables that field.
- Editing inputs clears the previous prediction. Click **Run Prediction** to
  obtain a result for the new inputs.

## Risk and counterfactual suggestions

The original threshold is retained: low risk is `p < 0.135`, and high risk is
`p >= 0.135`. Low-risk patients receive the risk result and SHAP explanations,
without counterfactual suggestions or a counterfactual scan.

For high-risk patients, single and combined suggestions must strictly decrease
the original model's predicted risk and satisfy all of these rules:

1. Preserve split-dose laxative use when already selected.
2. Keep dietary restriction duration within 0–2 days.
3. Preserve physical activity after laxative use when already selected.
4. Never decrease dietary restriction duration and PEG volume together, even
   if other factors also change.
5. Never decrease dietary restriction duration alone or PEG volume alone.
   Such a decrease can appear only with another actual changed factor.

Switching to Normal diet forces duration to 0. Those coupled changes do not
constitute two independent measures supporting a duration decrease; another
factor must change. Changing the same factor twice cannot disguise a single
change as a combination. Final plans are validated after all automatic changes
and deduplicated before scoring.

**Dose comparison:** the model distinguishes PEG 2 L, 3 L, and 4 L, so PEG volume
reductions can be compared directly. Sodium phosphate, mannitol, and magnesium
sulfate have no numeric dose field in this model. Their categorical codes are
not treated as equivalent doses or compared by integer magnitude.

The existing exclusion of fasting for one day or longer as a new diet/duration
target is retained. An unchanged baseline fasting plan does not prevent
suggestions about other factors.

The search considers single-factor actions and pairs of actions affecting
different factors. A Normal diet switch can also force duration to 0, so a
combined plan can contain three actual changed factors. The interface displays
the top 20 risk-decreasing plans and separate individual/combined tables.

## Files and checks

- `app.py`: interface and risk-dependent display.
- `model_utils.py`: model loading, inputs, encoding, and scaling.
- `clinical_rules.py`: BMI, duration, and final-plan constraints.
- `counterfactual_utils.py`: constrained search and scoring.
- `shap_utils.py`: original SHAP explanations, unchanged.

Run the focused input, search, and Streamlit interaction tests with:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

For an additional smoke test using the supplied fitted model:

```bash
python tests/native_model_smoke.py
```

The UI tests isolate the unchanged, expensive SHAP calculations; they test real
Streamlit inputs, risk display, and search branching. The separate native smoke
test loads the original model, checks prediction parity, and audits the emitted
counterfactual plans. It does not rerun the full global SHAP computation.

This application is intended for research and decision support and should not
replace clinician judgement or local clinical protocols.
