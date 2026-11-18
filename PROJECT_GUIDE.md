# PROJECT_GUIDE — Dengue Prediction (read this first)

## What is this project?
A machine-learning study on REAL data: can we predict Bangladesh's monthly dengue cases from weather? It follows a published 2025 paper (Rahman & Shiddik) and uses the same kind of real data they used — DGHS case counts and NASA climate data.

## The data (nothing is made up)
- Cases: DGHS monthly dengue cases, Jan 2001 – Dec 2024 (288 months, 660,976 cases in total). 2023 = 321,179 — the real record outbreak.
- Weather: NASA POWER satellite/reanalysis data for Dhaka — monthly temperature, humidity, rainfall.
- src/download_data.py fetches both and merges them; data/dengue_weather_monthly.csv is the merged file.

## Why the split matters
We train on 2001–2019 and test on 2020–2024. A random split would let the model peek at the future — for time series that is cheating (data leakage). The test period deliberately includes 2023, the hardest year.

## The two tasks
1. Regression — how many cases next month? Best: Random Forest, R² (log) = 0.6822. Linear Regression fails (R² < 0) because 2023 was unlike any training year — that failure is itself a finding.
2. Classification — will next month be an outbreak (≥5,000 cases)? Random Forest ROC-AUC = 0.9101, close to the published paper's 0.89.

## SHAP (explainable AI)
SHAP values show WHY the model predicts what it predicts. Top feature: cases_lag1. Rainfall 1–2 months earlier also matters — rain creates mosquito breeding sites, and dengue follows with a lag. See outputs/charts/shap_summary.png and shap_bar.png.

## Files
- src/analysis.py — EDA: time series, yearly totals, seasonality, rainfall-lag scatter.
- src/ml_model.py — both tasks + SHAP; writes regression_results.csv, classification_results.csv, summary.json.
- report/ — the full written report (same content as a small research paper).

## Viva questions
1. Why log-transform the target? — Case counts are extremely skewed (0 to 79,598); log makes errors comparable across quiet and outbreak months.
2. Why is a time-based split mandatory? — In deployment you only ever have the past; random splits leak the future.
3. Why did Linear Regression fail? — The relationship is nonlinear and 2023 exceeded the training range; tree models handle thresholds better.
4. What is SHAP? — A method that assigns each feature a contribution to each prediction, making the model explainable.
5. How could this be used in real life? — A 1–2 month early warning lets hospitals prepare beds and lets city corporations time mosquito-control drives.
6. Limitations? — National monthly totals hide district differences; early-2000s surveillance under-reported cases; weather is Dhaka-only while cases are national.
