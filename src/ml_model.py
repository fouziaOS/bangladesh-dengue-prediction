"""Machine learning on REAL dengue + climate data - extended version.

Pipeline
--------
1. Feature engineering: climate lags (rain 1-3 months), case lags (1, 12),
   seasonal month encoding.
2. Task 1 - regression of monthly cases (log target): SIX models compared.
3. TimeSeriesSplit cross-validation on the training period (no leakage).
4. Hyperparameter tuning of Random Forest with GridSearchCV (time-series CV).
5. Task 2 - outbreak classification (cases >= 5000): three classifiers.
6. SHAP explainability for the Random Forest regressor.
Split: train 2001-2019, test 2020-2024 (contains the record 2023 outbreak)."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                              RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, roc_auc_score, f1_score)

# ---------------------------------------------------------------- features
df = pd.read_csv("data/dengue_weather_monthly.csv", parse_dates=["year_month"])
df["month"] = df["year_month"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
for lag in (1, 2, 3):
    df[f"rain_lag{lag}"] = df["rainfall_mm_day"].shift(lag)
df["temp_lag1"] = df["temperature_c"].shift(1)
df["hum_lag1"] = df["humidity_pct"].shift(1)
df["cases_lag1"] = df["cases"].shift(1)
df["cases_lag12"] = df["cases"].shift(12)
df["log_cases"] = np.log1p(df["cases"])
d = df.dropna().reset_index(drop=True)
FEATS = ["temperature_c", "humidity_pct", "rainfall_mm_day", "rain_lag1", "rain_lag2",
         "rain_lag3", "temp_lag1", "hum_lag1", "cases_lag1", "cases_lag12",
         "month_sin", "month_cos"]
tr = d["year_month"] <= "2019-12-31"; te = ~tr
C = "outputs/charts/"; T = "outputs/tables/"
X, y = d[FEATS], d["log_cases"]

def reg_metrics(model, Xtr, ytr, Xte, yte_log, yte_raw):
    model.fit(Xtr, ytr)
    p_log = model.predict(Xte); p = np.expm1(p_log)
    return {"MAE (cases)": round(mean_absolute_error(yte_raw, p), 1),
            "RMSE (cases)": round(float(np.sqrt(mean_squared_error(yte_raw, p))), 1),
            "R2 (log scale)": round(r2_score(yte_log, p_log), 4)}, model, p

# ------------------------------------------------- Task 1: six regressors
MODELS = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
    "Support Vector Regression": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
    "Random Forest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
}
rows, fitted = [], {}
for name, m in MODELS.items():
    met, m, p = reg_metrics(m, X[tr], y[tr], X[te], y[te], d.loc[te, "cases"])
    rows.append({"Model": name, **met}); fitted[name] = (m, p)
res = pd.DataFrame(rows).sort_values("R2 (log scale)", ascending=False)
res.to_csv(T + "regression_results.csv", index=False)
print(res.to_string(index=False))

rf_model, rf_pred = fitted["Random Forest"]
plt.figure(figsize=(10, 4))
plt.plot(d.loc[te, "year_month"].values, d.loc[te, "cases"].values, label="Actual cases", color="#2471a3")
plt.plot(d.loc[te, "year_month"].values, rf_pred, label="Random Forest prediction", color="#c0392b", ls="--")
plt.ylabel("Monthly cases"); plt.legend()
plt.title("Test period 2020-2024: actual vs predicted (incl. the 2023 record outbreak)")
plt.tight_layout(); plt.savefig(C + "actual_vs_predicted.png", dpi=130); plt.close()

# --------------------------------------- cross-validation (training period)
tscv = TimeSeriesSplit(n_splits=5)
cv_rows = []
for name in ["Random Forest", "Gradient Boosting"]:
    scores = cross_val_score(MODELS[name], X[tr], y[tr], cv=tscv, scoring="r2", n_jobs=-1)
    cv_rows.append({"Model": name, "CV R2 mean": round(scores.mean(), 4), "CV R2 std": round(scores.std(), 4)})
pd.DataFrame(cv_rows).to_csv(T + "cv_results.csv", index=False)
print(pd.DataFrame(cv_rows).to_string(index=False))

# ------------------------------------------------------ RF hyperparameter tuning
grid = {"n_estimators": [300, 600], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2]}
gs = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), grid,
                  cv=TimeSeriesSplit(n_splits=3), scoring="r2", n_jobs=-1)
gs.fit(X[tr], y[tr])
tuned = gs.best_estimator_
met, _, _ = reg_metrics(tuned, X[tr], y[tr], X[te], y[te], d.loc[te, "cases"])
pd.DataFrame([{"Model": "Random Forest (tuned)", **met}]).to_csv(T + "tuned_results.csv", index=False)
print("best params:", gs.best_params_, "| tuned test:", met)

# ------------------------------------------------------------- SHAP (on RF)
expl = shap.TreeExplainer(rf_model)
sv = expl.shap_values(X[te])
shap.summary_plot(sv, X[te], show=False); plt.tight_layout()
plt.savefig(C + "shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()
mean_abs = np.abs(sv).mean(axis=0); order = np.argsort(mean_abs)
plt.figure(figsize=(8, 5)); plt.barh([FEATS[i] for i in order], mean_abs[order], color="#7d3c98")
plt.xlabel("Mean |SHAP value|"); plt.title("Which features drive dengue predictions? (SHAP)")
plt.tight_layout(); plt.savefig(C + "shap_bar.png", dpi=130); plt.close()

# ------------------------------------------- Task 2: outbreak classification
d["outbreak"] = (d["cases"] >= 5000).astype(int); yc = d["outbreak"]
CLFS = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42)}
rows2 = []
for name, m in CLFS.items():
    m.fit(X[tr], yc[tr]); pred = m.predict(X[te]); prob = m.predict_proba(X[te])[:, 1]
    rows2.append({"Model": name, "Accuracy": round(accuracy_score(yc[te], pred), 4),
                  "ROC-AUC": round(roc_auc_score(yc[te], prob), 4),
                  "F1": round(f1_score(yc[te], pred), 4)})
cls = pd.DataFrame(rows2)
cls.to_csv(T + "classification_results.csv", index=False)
print(cls.to_string(index=False))

summary = {"train_months": int(tr.sum()), "test_months": int(te.sum()),
           "outbreak_months_total": int(d["outbreak"].sum()),
           "top_shap_feature": FEATS[int(order[-1])],
           "total_cases": int(df["cases"].sum()),
           "cases_2023": int(df[df["year_month"].dt.year == 2023]["cases"].sum()),
           "best_regressor": res.iloc[0]["Model"], "best_r2": float(res.iloc[0]["R2 (log scale)"]),
           "tuned_rf": met, "best_params": gs.best_params_}
json.dump(summary, open("outputs/summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, default=str))
