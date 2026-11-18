"""Data sources (both REAL, public):
1. Dengue cases: monthly national cases 2001-2024 compiled from DGHS (Directorate
   General of Health Services) press releases by the DENV-Data-Analysis project;
   file bd_dengue_monthly.csv (mirrored in the SasthoSetu repository, data/real/).
2. Weather: NASA POWER monthly API for Dhaka (lat 23.8103, lon 90.4125):
   T2M (temperature, deg C), RH2M (relative humidity, %), PRECTOTCORR (rainfall, mm/day).
This script merges them into data/dengue_weather_monthly.csv (288 months).
The merged CSV is committed so the project also runs offline."""
import io, json, urllib.request
import pandas as pd

CASES_URL = "https://raw.githubusercontent.com/meherabmehu/SasthoSetu/HEAD/data/real/bd_dengue_monthly.csv"
NASA_URL = ("https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,RH2M,PRECTOTCORR"
            "&community=AG&longitude=90.4125&latitude=23.8103&start=2001&end=2024&format=JSON")

cases = pd.read_csv(io.StringIO(urllib.request.urlopen(CASES_URL).read().decode()))
raw = json.loads(urllib.request.urlopen(NASA_URL).read())["properties"]["parameter"]
rows = {}
for key, name in [("T2M", "temperature_c"), ("RH2M", "humidity_pct"), ("PRECTOTCORR", "rainfall_mm_day")]:
    for ym, v in raw[key].items():
        y, m = int(ym[:4]), int(ym[4:6])
        if m == 13:  # NASA month 13 = annual average, skip
            continue
        rows.setdefault(f"{y}-{m:02d}", {})[name] = v
weather = pd.DataFrame([{"year_month": k, **v} for k, v in rows.items()])
df = cases.merge(weather, on="year_month", how="inner").sort_values("year_month")
df.to_csv("data/dengue_weather_monthly.csv", index=False)
print("saved", len(df), "months:", df["year_month"].iloc[0], "->", df["year_month"].iloc[-1])
