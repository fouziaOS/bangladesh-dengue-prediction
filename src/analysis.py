"""Exploratory analysis of REAL dengue + weather data (2001-2024)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/dengue_weather_monthly.csv", parse_dates=["year_month"])
df["year"] = df["year_month"].dt.year; df["month"] = df["year_month"].dt.month
C = "outputs/charts/"; T = "outputs/tables/"

yearly = df.groupby("year")["cases"].sum()
yearly.rename("total_cases").to_csv(T + "yearly_cases.csv")
season = df.groupby("month")["cases"].agg(["mean", "sum"]).round(1)
season.columns = ["avg_cases", "total_cases"]; season.to_csv(T + "monthly_seasonality.csv")
df.corr(numeric_only=True).round(3).to_csv(T + "correlation_matrix.csv")

plt.figure(figsize=(10, 4)); plt.plot(df["year_month"], df["cases"], color="#c0392b", lw=1)
plt.ylabel("Monthly cases"); plt.title("Monthly dengue cases in Bangladesh, 2001-2024 (DGHS data)")
plt.tight_layout(); plt.savefig(C + "cases_timeseries.png", dpi=130); plt.close()

plt.figure(figsize=(9, 4.5)); yearly.plot(kind="bar", color="#7d3c98")
plt.ylabel("Total cases"); plt.title("Total dengue cases per year - 2023 was the record outbreak")
plt.tight_layout(); plt.savefig(C + "yearly_cases.png", dpi=130); plt.close()

plt.figure(figsize=(8, 4.5)); season["avg_cases"].plot(kind="bar", color="#2471a3")
plt.ylabel("Average monthly cases"); plt.xlabel("Month"); plt.title("Seasonality: cases peak after the monsoon (Aug-Oct)")
plt.tight_layout(); plt.savefig(C + "cases_by_month.png", dpi=130); plt.close()

d = df.copy(); d["rain_lag2"] = d["rainfall_mm_day"].shift(2)
plt.figure(figsize=(7, 4.5)); plt.scatter(d["rain_lag2"], d["cases"], s=10, alpha=0.45, color="#1e8449")
plt.xlabel("Rainfall 2 months earlier (mm/day)"); plt.ylabel("Dengue cases")
plt.title("Rainfall leads dengue by ~2 months (mosquito breeding lag)")
plt.tight_layout(); plt.savefig(C + "rain_lag_scatter.png", dpi=130); plt.close()
print("analysis done; total cases:", int(df["cases"].sum()), "| 2023:", int(yearly.get(2023, 0)))
