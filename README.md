# Predicting Dengue Outbreaks in Bangladesh from Climate Data — A Machine Learning Study on Real DGHS & NASA Data

## Abstract
Dengue is Bangladesh's most dangerous seasonal epidemic — the 2023 outbreak alone recorded **321,179 cases** in the DGHS-compiled data used here. This study asks whether monthly dengue cases can be predicted from climate variables, following Rahman & Shiddik (2025), who reported XGBoost AUC 0.89 for outbreak prediction. Using **real data only** — monthly DGHS dengue cases (2001–2024, 288 months) merged with NASA POWER weather for Dhaka (temperature, humidity, rainfall) — we train regression and classification models with an honest **time-based split** (train 2001–2019, test 2020–2024). Random Forest predicts monthly cases with **R² = 0.6822** (log scale) and detects outbreak months (≥5,000 cases) with **ROC-AUC = 0.9101**, comparable to the published study. SHAP analysis shows last month's case count and lagged rainfall are the strongest signals — supporting a practical **1–2 month early-warning** use case.

## Data (real, public)
| Source | What | Period |
|---|---|---|
| DGHS via DENV-Data-Analysis compilation | Monthly national dengue cases | 2001–2024 (288 months, 660,976 total cases) |
| NASA POWER API (Dhaka, 23.81°N 90.41°E) | Monthly temperature, humidity, rainfall | 2001–2024 |

No synthetic data is used anywhere in this project. `src/download_data.py` re-fetches and merges both sources.

## Method
- Features: temperature, humidity, rainfall; rainfall lags 1–3 months; temperature/humidity lag 1; cases lag 1 and lag 12; seasonal (month) sine/cosine.
- Task 1 — regression of monthly cases (log-transformed target): Linear Regression, Random Forest, Gradient Boosting.
- Task 2 — classification of outbreak months (cases ≥ 5,000): Logistic Regression, Random Forest.
- Evaluation: time-based split only (train 216 months, test 60 months) — random splits are not valid for time series.
- Explainability: SHAP (TreeExplainer) on the Random Forest regressor.

## Results
### Task 1 — predicting monthly cases (test: 2020–2024)
| Model | MAE (cases) | RMSE (cases) | R² (log scale) |
|---|---|---|---|
| Linear Regression | 55559.0 | 278893.4 | -0.0699 |
| Random Forest | 6886.1 | 16509.4 | 0.6822 |
| Gradient Boosting | 7627.2 | 18233.2 | 0.5915 |

### Task 2 — predicting outbreak months (test: 2020–2024)
| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.7833 | 0.8883 | 0.6667 |
| Random Forest | 0.7667 | 0.9101 | 0.4167 |

Linear Regression fails on the regression task (R² < 0) because the 2023 outbreak was far larger than anything in the training years — a real-data difficulty that synthetic data would never show. Tree models cope much better. Our outbreak AUC (0.9101) is in the same range as Rahman & Shiddik's XGBoost (0.89).

## Key findings
- Dengue in Bangladesh is strongly seasonal: cases peak in **August–October**, after the monsoon rains.
- Rainfall leads cases by about **2 months** (mosquito breeding lag) — visible in `outputs/charts/rain_lag_scatter.png`.
- SHAP: the top driver is **cases_lag1** (last month's cases), followed by lagged rainfall — outbreaks build momentum and are climate-primed.
- 2023 (321,179 cases) was ~3× worse than the next-worst years (2019: 101,354; 2024: 101,214).

## Files
```
src/download_data.py   fetches + merges the two real datasets
src/analysis.py        EDA (4 charts, 3 tables)
src/ml_model.py        regression + classification + SHAP (3 charts, 2 tables)
report/                the full written report (PDF + DOCX)
```

## Run it
```bash
pip install -r requirements.txt
python src/download_data.py   # or use the committed CSV in data/
python src/analysis.py && python src/ml_model.py
```

## References
1. Rahman, M.S. & Shiddik, M.A.B. (2025). Climate-driven dengue prediction in Bangladesh. *Global Epidemiology*, 10, 100210.
2. Shiddik, M.A.B. et al. (2026). Spatiotemporal dengue forecasting across 64 districts. *Tropical Medicine and Infectious Disease*, 11, 73.
3. Hossain, M.S. et al. — DGHS HEOC dengue dashboard data, 2018–2025.
4. NASA POWER Project — monthly meteorological data, Dhaka.
