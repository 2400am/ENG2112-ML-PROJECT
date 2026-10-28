"""Step 6: figures for the report.

Reads the outputs of steps 2-5 and saves PNG figures to outputs/figures/.
Run after 05_spatial.py.
"""
import matplotlib
matplotlib.use("Agg")                     # save to files, don't open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import config as c

FIG = c.OUTPUTS / "figures"
FIG.mkdir(exist_ok=True)

# One consistent look for every figure: thin marks, light grid, no top/right box lines.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", FIG / name)


daily = pd.read_pickle(c.PROCESSED / "pm25_daily.pkl")
sites = pd.read_csv(c.PROCESSED / "sites.csv")
coverage = pd.read_csv(c.OUTPUTS / "station_coverage.csv")

# ===================== 1. station map, coloured by data coverage =====================
s = sites.merge(coverage[["site_id", "valid_days"]], on="site_id")
fig, ax = plt.subplots(figsize=(6.5, 6))
b = c.BBOX
ax.add_patch(plt.Rectangle((b["lon_min"], b["lat_min"]), b["lon_max"] - b["lon_min"],
                           b["lat_max"] - b["lat_min"], fill=False, ec=INK2, ls="--", lw=1))
pts = ax.scatter(s["longitude"], s["latitude"], c=s["valid_days"], cmap="Blues", vmin=0,
                 s=70, ec="white", lw=1.5, zorder=3)
for _, r in s.iterrows():
    ax.annotate(r["site_name"].title(), (r["longitude"], r["latitude"]), xytext=(5, 4),
                textcoords="offset points", fontsize=7, color=INK)
fig.colorbar(pts, ax=ax, shrink=0.7, label="Valid days (2016-2024)")
ax.set(xlabel="Longitude", ylabel="Latitude", aspect=1 / np.cos(np.radians(-33.9)),
       title=f"{len(s)} PM2.5 monitoring stations in the Greater Sydney study box")
save(fig, "1_station_map.png")

# ========== 2. Sydney-wide daily PM2.5 over time, with the WHO guideline ==========
city = daily.groupby("date")["pm25_mean"].mean()
cap = 60                                  # y-axis limit so everyday levels are visible
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axvspan(pd.Timestamp(c.FIRE_START), pd.Timestamp(c.FIRE_END), color=SHADE, zorder=0)
ax.plot(city.index, city.clip(upper=cap).values, color=BLUE, lw=0.8)
ax.axhline(c.WHO_24H, color=INK2, ls="--", lw=1)
ax.text(pd.Timestamp("2016-02-01"), c.WHO_24H + 2, f"WHO 24-h guideline ({c.WHO_24H:g} µg/m³)",
        fontsize=8, color=INK2)
ax.text(pd.Timestamp(c.FIRE_START), cap * 0.93,
        f" 2019-20 bushfires\n peak {city.max():.0f} µg/m³ (off scale)", fontsize=8, color=INK2, va="top")
for year, label in [(c.VAL_YEAR, "val"), (c.TEST_YEAR, "test")]:
    ax.axvline(pd.Timestamp(f"{year}-01-01"), color=INK2, lw=0.8)
    ax.text(pd.Timestamp(f"{year}-01-01"), cap * 0.93, f" {label}", fontsize=8, color=INK2, va="top")
ax.set(ylabel="PM2.5 (µg/m³)", ylim=(0, cap),
       title="Sydney-wide daily mean PM2.5 (average of all stations), 2016-2024")
save(fig, "2_pm25_timeseries.png")

# ===================== 3. forecast: precision-recall curves (test year) =====================
preds = pd.read_pickle(c.PROCESSED / "forecast_preds.pkl")
test = preds[preds["split"] == "test"]
names = {"gbm": "Gradient boosting", "logreg": "Logistic regression",
         "persistence": "Persistence (today's PM2.5)", "climatology": "Climatology"}
fig, ax = plt.subplots(figsize=(6.5, 4.8))
for m, colour in zip(["gbm", "logreg", "persistence", "climatology"], [BLUE, ORANGE, AQUA, YELLOW]):
    score = test[m].fillna(0)
    p, r, _ = precision_recall_curve(test["y"], score)
    ax.plot(r, p, color=colour, label=f"{names[m]} (PR-AUC {average_precision_score(test['y'], score):.2f})")
ax.axhline(test["y"].mean(), color=INK2, ls="--", lw=1)
ax.text(0.01, test["y"].mean() + 0.015, f"random guessing ({test['y'].mean():.2f})",
        fontsize=8, color=INK2)
ax.set(xlabel="Recall (share of exceedance days caught)", ylabel="Precision (share of alerts correct)",
       xlim=(0, 1), ylim=(0, 1), title=f"Next-day WHO exceedance forecast, {c.TEST_YEAR} test year")
ax.legend(loc="upper right", fontsize=8)
save(fig, "3_forecast_pr_curves.png")

# ===================== 4. forecast: which features matter =====================
imp = pd.read_csv(c.OUTPUTS / "forecast_feature_importance.csv").sort_values("PR_AUC_drop")
labels = {"pm_today": "Own PM2.5 today (to 4 pm)", "pm_today_max": "Own PM2.5 today, max hour",
          "pm_last3h": "Own PM2.5, 1-4 pm", "pm_yesterday": "Own PM2.5 yesterday",
          "pm_3day": "Own PM2.5, last 3 days", "pm_7day": "Own PM2.5, last 7 days",
          "net_today": "Sydney-wide PM2.5 today", "net_yesterday": "Sydney-wide PM2.5 yesterday",
          "wx_temp": "Temperature", "wx_rh": "Humidity", "wx_wind": "Wind speed",
          "wx_wind_u": "Wind, east-west", "wx_wind_v": "Wind, north-south",
          "wx_blh": "Boundary layer height", "wx_rain": "Rain",
          "doy_sin": "Season (sin)", "doy_cos": "Season (cos)", "weekend_tomorrow": "Tomorrow is a weekend"}
imp["feature"] = imp["feature"].map(labels)
fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.barh(imp["feature"], imp["PR_AUC_drop"], xerr=imp["std"], color=BLUE, height=0.6,
        error_kw=dict(ecolor=INK2, lw=0.8, capsize=0))
ax.axvline(0, color=INK2, lw=0.8)
ax.grid(axis="y", visible=False)
ax.set(xlabel="Drop in PR-AUC when the feature is shuffled (bigger = more important)",
       title="Gradient boosting forecast: feature importance (test year)")
save(fig, "4_forecast_feature_importance.png")

# ===================== 5. spatial: error vs distance to nearest station =====================
dist = pd.read_csv(c.OUTPUTS / "spatial_error_by_distance.csv")
titles = {"S1_LOSO_all_years": "S1: one station hidden, all years",
          "S2_LOSO_test_year": f"S2: one station hidden, {c.TEST_YEAR}",
          "S3_cluster_out_test_year": f"S3: cluster hidden, {c.TEST_YEAR}"}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (test_name, title) in zip(axes, titles.items()):
    d = dist[(dist["test"] == test_name) & (dist["n"] >= 30)]   # skip bands with too few days
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["idw_RMSE"], width=0.38, color=ORANGE, label="IDW (baseline)")
    ax.bar(x + 0.2, d["gbm_RMSE"], width=0.38, color=BLUE, label="Gradient boosting")
    ax.set_xticks(x, d["dist_band"])
    ax.grid(axis="x", visible=False)
    ax.set(title=title, xlabel="Distance to nearest station")
axes[0].set_ylabel("RMSE (µg/m³)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8,
           bbox_to_anchor=(0.5, -0.15))
fig.suptitle("Spatial estimation error vs distance to the nearest station", x=0.01, y=1.06,
             ha="left", fontweight="bold", fontsize=11)
save(fig, "5_spatial_error_by_distance.png")

# ===================== 6. spatial: estimated vs measured (S2, test year) =====================
s2 = pd.read_pickle(c.PROCESSED / "spatial_preds_S2_LOSO_test_year.pkl")
fig, ax = plt.subplots(figsize=(5, 5))
top = np.nanpercentile(np.r_[s2["pm25_mean"], s2["idw_baseline"]], 99.9)
ax.scatter(s2["pm25_mean"], s2["idw_baseline"], s=8, color=BLUE, alpha=0.25, lw=0)
ax.plot([0, top], [0, top], color=INK2, lw=1, ls="--")
ax.text(top * 0.97, top * 0.90, "perfect estimate", ha="right", fontsize=8, color=INK2)
ax.set(xlim=(0, top), ylim=(0, top), xlabel="Measured daily PM2.5 (µg/m³)",
       ylabel="IDW estimate from other stations (µg/m³)",
       title=f"Hidden-station estimates, {c.TEST_YEAR}")
save(fig, "6_spatial_estimated_vs_measured.png")

# ===================== 7. spatial: where estimates can be trusted =====================
G = pd.read_csv(c.OUTPUTS / "grid_area_of_applicability.csv")
inside = G.pivot_table(index="latitude", columns="longitude", values="inside_tested_range")
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.pcolormesh(inside.columns, inside.index, inside.values.astype(float),
              cmap=matplotlib.colors.ListedColormap([GRID, "#cfe0f6"]), vmin=0, vmax=1, shading="nearest")
ax.scatter(sites["longitude"], sites["latitude"], s=40, color=BLUE, ec="white", lw=1.2, zorder=3)
share = G["inside_tested_range"].mean()
handles = [plt.Rectangle((0, 0), 1, 1, color="#cfe0f6"), plt.Rectangle((0, 0), 1, 1, color=GRID),
           plt.Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=7)]
ax.legend(handles, [f"Within tested distance ({share:.0%} of cells)", "Outside tested range",
                    "Monitoring station"], loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
ax.grid(False)
ax.set(xlabel="Longitude", ylabel="Latitude", aspect=1 / np.cos(np.radians(-33.9)),
       title="Area of applicability on the 1 km grid")
save(fig, "7_spatial_area_of_applicability.png")