"""Step 5: estimate daily PM2.5 at a place with no monitor.

For each station in turn, pretend it doesn't exist: estimate its daily PM2.5
using ONLY the other stations, plus weather and time of year at its location,
then compare the estimate with what it actually measured.

Three tests, from easiest to hardest:
  S1  leave-one-station-out, all years
  S2  leave-one-station-out, trained on 2016-2022, tested on 2024
  S3  leave-CLUSTER-out (a whole group of nearby stations hidden), tested on 2024
      - the target is further from any remaining station, closer to a truly unmonitored suburb

Reads   data/processed/pm25_daily.pkl, sites.csv;  data/raw/weather_hourly.csv
Writes  outputs/spatial_results.csv               errors for every model, test and stratum
        outputs/spatial_error_by_distance.csv     error vs distance to the nearest station
        outputs/grid_area_of_applicability.csv    which 1 km cells are within the tested distance
        data/processed/spatial_preds_<test>.pkl   every prediction, for maps and plots
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import config as c

# ================================ load ================================
daily = pd.read_pickle(c.PROCESSED / "pm25_daily.pkl")
sites = pd.read_csv(c.PROCESSED / "sites.csv").set_index("site_id")
wx = pd.read_csv(c.RAW / "weather_hourly.csv", parse_dates=["time"])
wx = wx[wx["site_id"].isin(sites.index)]

# Daily weather at each station. Wind as east-west (u) and north-south (v) parts.
wx["date"] = wx["time"].dt.normalize()
rad = np.deg2rad(wx["wind_direction_10m"])
wx["u"] = -wx["wind_speed_10m"] * np.sin(rad)
wx["v"] = -wx["wind_speed_10m"] * np.cos(rad)
wxd = (wx.groupby(["site_id", "date"])
         .agg(wx_temp=("temperature_2m", "mean"), wx_rh=("relative_humidity_2m", "mean"),
              wx_wind=("wind_speed_10m", "mean"), wx_blh=("boundary_layer_height", "mean"),
              wx_rain=("precipitation", "sum"), wx_u=("u", "mean"), wx_v=("v", "mean"))
         .reset_index())
WIND = wxd.set_index(["site_id", "date"])[["wx_u", "wx_v"]]

# Wide table: one row per day, one column per station (NaN = no valid reading that day).
P = daily.pivot(index="date", columns="site_id", values="pm25_mean")
ALL = list(P.columns)

FEATURES = ["idw", "nearest_val", "nearest_km", "mean3", "net_mean", "net_std", "n_donors",
            "upwind_idw", "upwind_minus_idw", "doy_sin", "doy_cos",
            "wx_temp", "wx_rh", "wx_wind", "wx_blh", "wx_rain", "wx_u", "wx_v"]
# Latitude/longitude are deliberately NOT features: with only ~19 stations, a model
# would use them to memorise which station is which, and fail at new places.


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between points on the Earth (haversine formula)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def site_features(s, donors):
    """Features for target station s on every day, built ONLY from the `donors` stations."""
    donors = [d for d in donors if d != s]
    lat0, lon0 = sites.loc[s, "latitude"], sites.loc[s, "longitude"]
    dist = pd.Series(distance_km(lat0, lon0, sites.loc[donors, "latitude"].values,
                                 sites.loc[donors, "longitude"].values), index=donors)
    V = P[donors]                       # donor readings: days x donors
    avail = V.notna()

    # Inverse-distance weighting (IDW): closer stations count more (weight = 1 / distance^2).
    w = avail.mul(1 / np.maximum(dist, 0.5) ** 2)
    idw = (V.fillna(0) * w).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)

    # Nearest station that reported that day, and the mean of the 3 nearest that reported.
    Vo = V[dist.sort_values().index]
    nearest_val = Vo.bfill(axis=1).iloc[:, 0]
    nearest_id = Vo.notna().idxmax(axis=1)
    take = Vo.notna() & (Vo.notna().cumsum(axis=1) <= 3)
    mean3 = Vo.where(take).sum(axis=1) / take.sum(axis=1).replace(0, np.nan)

    # Upwind IDW: like IDW, but stations the wind is blowing FROM count more that day
    # (air arriving here has just passed over them). 1 = straight upwind, 0 = side-on or downwind.
    dx = (sites.loc[donors, "longitude"].values - lon0) * np.cos(np.radians(lat0))   # east
    dy = sites.loc[donors, "latitude"].values - lat0                                  # north
    bearing = np.arctan2(dx, dy)                                     # direction to each donor
    wind = WIND.loc[s].reindex(P.index)
    wind_from = np.arctan2(-wind["wx_u"].values, -wind["wx_v"].values)
    align = np.clip(np.cos(bearing[None, :] - wind_from[:, None]), 0, None)
    wu = avail.values * align / np.maximum(dist.values, 0.5) ** 2
    total = wu.sum(axis=1)
    upwind = np.where(total > 0, (V.fillna(0).values * wu).sum(axis=1) / np.where(total > 0, total, 1), np.nan)
    upwind = pd.Series(upwind, index=P.index).fillna(idw)           # nobody upwind -> use IDW

    f = pd.DataFrame({"date": P.index, "site_id": s,
                      "idw": idw.values, "nearest_val": nearest_val.values,
                      "nearest_km": dist.reindex(nearest_id.values).values, "mean3": mean3.values,
                      "net_mean": V.mean(axis=1).values, "net_std": V.std(axis=1).values,
                      "n_donors": avail.sum(axis=1).values,
                      "upwind_idw": upwind.values, "upwind_minus_idw": (upwind - idw).values})
    doy = f["date"].dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f = f.merge(daily[["site_id", "date", "pm25_mean", "fire_period", "local_spike"]],
                on=["site_id", "date"])                              # keep days with a true value
    return f.merge(wxd, on=["site_id", "date"], how="left")


GBM_SETTINGS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                    l2_regularization=1.0, random_state=c.SEED)


def run_fold(hidden, train_rows, test_rows):
    """Hide `hidden` stations completely (never donors, never training), train on the rest,
    and estimate the hidden ones."""
    network = [s for s in ALL if s not in hidden]
    tr = pd.concat([site_features(s, network) for s in network])
    te = pd.concat([site_features(s, network) for s in hidden])
    tr, te = tr[train_rows(tr)].dropna(subset=["idw"]), te[test_rows(te)].dropna(subset=["idw"])

    # The GBM learns a CORRECTION to IDW (as a log ratio), not the level itself.
    # Trees can't predict beyond values seen in training, so on extreme smoke days a
    # level model fails; a correction to an IDW estimate that already tracks the smoke doesn't.
    y = np.log1p(tr["pm25_mean"]) - np.log1p(tr["idw"].clip(lower=0))
    base = np.log1p(te["idw"].clip(lower=0))
    model = HistGradientBoostingRegressor(**GBM_SETTINGS).fit(tr[FEATURES], y)
    pred_log = base + model.predict(te[FEATURES])

    # 80% prediction interval: how wrong is the model at a station it never saw?
    # Leave whole training stations out, collect those errors, take the 10th-90th percentile.
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(n_splits=min(5, tr["site_id"].nunique())).split(tr, y, tr["site_id"]):
        m = HistGradientBoostingRegressor(**GBM_SETTINGS).fit(tr[FEATURES].iloc[a], y.iloc[a])
        oof[b] = m.predict(tr[FEATURES].iloc[b])
    err_lo, err_hi = np.quantile(y.values - oof, [0.10, 0.90])

    out = te.copy()
    out["gbm"] = np.expm1(pred_log)
    out["q10"], out["q90"] = np.expm1(pred_log + err_lo), np.expm1(pred_log + err_hi)
    out["idw_baseline"], out["nearest_baseline"], out["netmean_baseline"] = te["idw"], te["nearest_val"], te["net_mean"]
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


# ============================== run the tests ==============================
is_train = lambda d: d["date"] <= c.TRAIN_END
is_test = lambda d: d["date"].dt.year.astype(str) == c.TEST_YEAR
any_day = lambda d: np.ones(len(d), bool)

km = KMeans(c.N_SPATIAL_CLUSTERS, n_init=10, random_state=c.SEED).fit(sites[["latitude", "longitude"]])
cluster = pd.Series(km.labels_, index=sites.index)

tests = {}
print("S1: leave-one-station-out, all years...")
tests["S1_LOSO_all_years"] = pd.concat([run_fold([s], any_day, any_day) for s in ALL])
print("S2: leave-one-station-out, train 2016-2022, test 2024...")
tests["S2_LOSO_test_year"] = pd.concat([run_fold([s], is_train, is_test) for s in ALL])
print("S3: leave-cluster-out, train 2016-2022, test 2024...")
tests["S3_cluster_out_test_year"] = pd.concat(
    [run_fold(list(cluster.index[cluster == k]), is_train, is_test) for k in range(c.N_SPATIAL_CLUSTERS)])

# ================================ score ================================
MODELS = ["gbm", "idw_baseline", "nearest_baseline", "netmean_baseline"]
rng = np.random.default_rng(c.SEED)
rows, by_dist = [], []
for name, R in tests.items():
    R.to_pickle(c.PROCESSED / f"spatial_preds_{name}.pkl")
    strata = {"all": R, "fire_2019_20": R[R["fire_period"]], "non_fire": R[~R["fire_period"]],
              "excl_local_spikes": R[~R["local_spike"]]}
    for stratum, r in strata.items():
        if len(r) == 0 or (stratum != "all" and len(r) == len(R)):
            continue                    # empty, or identical to "all"
        for m in MODELS:
            row = {"test": name, "stratum": stratum, "model": m, "n": len(r),
                   "MAE": mae(r["pm25_mean"], r[m]), "RMSE": rmse(r["pm25_mean"], r[m]),
                   "R2": r2(r["pm25_mean"], r[m])}
            if m == "gbm":
                row["PI80_coverage"] = float(r["pm25_mean"].between(r["q10"], r["q90"]).mean())
            rows.append(row)

        # Is the GBM really better than the best baseline? Resample whole STATIONS.
        best = min(MODELS[1:], key=lambda m: rmse(r["pm25_mean"], r[m]))
        reduction = lambda d: 1 - rmse(d["pm25_mean"], d["gbm"]) / rmse(d["pm25_mean"], d[best])
        groups = {s: g for s, g in r.groupby("site_id")}
        ids = list(groups)
        boot = [reduction(pd.concat([groups[s] for s in rng.choice(ids, len(ids))])) for _ in range(c.N_BOOT)]
        point, (lo, hi) = reduction(r), np.percentile(boot, [2.5, 97.5])
        rows.append({"test": name, "stratum": stratum, "model": f"RMSE reduction vs {best}",
                     "RMSE": point, "CI_low": lo, "CI_high": hi,
                     "meets_criterion": bool(point >= c.MIN_RMSE_REDUCTION and lo > 0)})

    # Error vs distance to the nearest station that reported that day.
    R = R.assign(dist_band=pd.cut(R["nearest_km"], [0, 5, 10, 20, 40, 1e9],
                                  labels=["0-5 km", "5-10 km", "10-20 km", "20-40 km", "40+ km"]))
    d = R.groupby("dist_band", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "gbm_RMSE": rmse(g["pm25_mean"], g["gbm"]),
        "idw_RMSE": rmse(g["pm25_mean"], g["idw_baseline"])}), include_groups=False).reset_index()
    by_dist.append(d.assign(test=name))

results = pd.DataFrame(rows)
results.to_csv(c.OUTPUTS / "spatial_results.csv", index=False)
dist_table = pd.concat(by_dist)
dist_table.to_csv(c.OUTPUTS / "spatial_error_by_distance.csv", index=False)

# =================== area of applicability on a 1 km grid ===================
# Which parts of Sydney are no further from a station than the places we actually tested?
b = c.BBOX
lat = np.arange(b["lat_min"], b["lat_max"], c.GRID_KM / 111)
lon = np.arange(b["lon_min"], b["lon_max"], c.GRID_KM / (111 * np.cos(np.radians(-33.9))))
G = pd.DataFrame([(a, o) for a in lat for o in lon], columns=["latitude", "longitude"])
G["nearest_station_km"] = np.min([distance_km(G["latitude"], G["longitude"], r["latitude"], r["longitude"])
                                  for _, r in sites.iterrows()], axis=0)
max_tested = tests["S3_cluster_out_test_year"]["nearest_km"].quantile(0.95)
G["inside_tested_range"] = G["nearest_station_km"] <= max_tested
G.to_csv(c.OUTPUTS / "grid_area_of_applicability.csv", index=False)

# ================================ report ================================
pd.set_option("display.width", 220)
for (name, stratum), r in results.groupby(["test", "stratum"], sort=False):
    print(f"\n{name} - {stratum}")
    print(r.drop(columns=["test", "stratum"]).round(3).to_string(index=False))
print("\nError by distance to the nearest station\n", dist_table.round(2).to_string(index=False))
print(f"\n{G['inside_tested_range'].mean():.0%} of the 1 km grid cells are within the tested distance "
      f"({max_tested:.1f} km of a station); the rest should be marked 'outside tested range' on any map.")