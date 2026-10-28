"""
build forecast table

each row is one station on one day D, as seen at 16:00 on D
features use only data available by then and the target is whether day D+1's deaily mean exceeds the WHO guidelines

reads   data/processed/pm25_hourly_clean.pkl, pm25_daily.pkl
        data/raw/weather_hourly.csv
writes  data/processed/forecast_table.pkl
"""

import numpy as np
import pandas as pd
import config as c

FEATURES = c.FEATURES

def build_features(pm, wx, daily):
    """ one row per station per day D, using only data known at the cutoff on D."""
    cut = c.FORECAST_CUTOFF_HOUR

    # grid: every station x every calendar day + missing days become Nan
    sites = sorted(daily["site_id"].unique())
    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([sites, dates], names=["site_id", "date"])
    t = pd.DataFrame(index=grid)
    day_of = t.index.get_level_values("date")          # the date of each row

    # past days at this station
    s = daily.set_index(["site_id", "date"])["pm25_mean"].reindex(grid)
    by_site = s.groupby(level="site_id")
    t["pm_yesterday"] = by_site.shift(1)
    t["pm_3day"] = by_site.transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    t["pm_7day"] = by_site.transform(lambda x: x.shift(1).rolling(7, min_periods=4).mean())

    # today up to 4pm
    pm = pm.assign(date=pm["time"].dt.normalize(), hour=pm["time"].dt.hour)
    before = pm[pm["hour"] < cut]
    t = t.join(before.groupby(["site_id", "date"])["pm25"].agg(pm_today="mean", pm_today_max="max"))
    last3 = before[before["hour"] >= cut - 3]
    t = t.join(last3.groupby(["site_id", "date"])["pm25"].mean().rename("pm_last3h"))

    # the whole network
    net_today = before.groupby("date")["pm25"].mean()
    net_daily = s.groupby(level="date").mean()
    t["net_today"] = net_today.reindex(day_of).values
    t["net_yesterday"] = net_daily.shift(1).reindex(day_of).values

    # weather at this station to 4pm
    wx = wx.assign(date=wx["time"].dt.normalize(), hour=wx["time"].dt.hour)
    wb = wx[wx["hour"] < cut].copy()
    rad = np.deg2rad(wb["wind_direction_10m"])
    wb["wind_u"] = -wb["wind_speed_10m"] * np.sin(rad)   # + = air moving east
    wb["wind_v"] = -wb["wind_speed_10m"] * np.cos(rad)   # + = air moving north
    t = t.join(wb.groupby(["site_id", "date"]).agg(
        wx_temp=("temperature_2m", "mean"),
        wx_rh=("relative_humidity_2m", "mean"),
        wx_wind=("wind_speed_10m", "mean"),
        wx_wind_u=("wind_u", "mean"),
        wx_wind_v=("wind_v", "mean"),
        wx_blh=("boundary_layer_height", "mean"),
        wx_rain=("precipitation", "sum"),
    ))

    # calendar
    doy = day_of.dayofyear
    t["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    t["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    t["weekend_tomorrow"] = ((day_of + pd.Timedelta(days=1)).dayofweek >= 5).astype(int)

    # target and labels, all about tmr
    t["target_pm"] = by_site.shift(-1)
    t["target"] = np.where(t["target_pm"].notna(), t["target_pm"] > c.WHO_24H, np.nan)
    labels = daily.set_index(["site_id", "date"])[["split", "fire_period", "local_spike"]]
    tomorrow = pd.MultiIndex.from_arrays([t.index.get_level_values("site_id"), day_of + pd.Timedelta(days=1)])
    t[["split", "fire_period", "local_spike"]] = labels.reindex(tomorrow).values

    return t.reset_index()


# build
pm = pd.read_pickle(c.PROCESSED / "pm25_hourly_clean.pkl")
daily = pd.read_pickle(c.PROCESSED / "pm25_daily.pkl")
wx = pd.read_csv(c.RAW / "weather_hourly.csv", parse_dates=["time"])
wx = wx[wx["site_id"].isin(daily["site_id"].unique())]      # same 19 stations

table = build_features(pm, wx, daily)
table = table.dropna(subset=["target"])                      # need a known answer
table.to_pickle(c.PROCESSED / "forecast_table.pkl")

print(f"{len(table):,} rows (station-days with a known target)")
print(table.groupby("split")["target"].agg(rows="size", exceed_rate="mean").round(3))
print("\nShare of rows missing each feature:")
print(table[FEATURES].isna().mean().round(3).to_string())

# leakage check
D = pd.Timestamp("2021-06-15")
lo, hi = D - pd.Timedelta(days=10), D + pd.Timedelta(days=2)
pm_w = pm[pm["time"].between(lo, hi)].copy()
wx_w = wx[wx["time"].between(lo, hi)].copy()
daily_w = daily[daily["date"].between(lo, hi)].copy()

cutoff = D + pd.Timedelta(hours=c.FORECAST_CUTOFF_HOUR)
rng = np.random.default_rng(0)
pm_x, wx_x, daily_x = pm_w.copy(), wx_w.copy(), daily_w.copy()
late = pm_x["time"] >= cutoff
pm_x.loc[late, "pm25"] = rng.uniform(0, 500, late.sum())
late = wx_x["time"] >= cutoff
for col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "precipitation", "boundary_layer_height"]:
    wx_x[col] = wx_x[col].astype(float)
    wx_x.loc[late, col] = rng.uniform(0, 1000, late.sum())
late = daily_x["date"] >= D                  # today's full day mean isn't known at 16:00 either
daily_x.loc[late, "pm25_mean"] = rng.uniform(0, 500, late.sum())

a = build_features(pm_w, wx_w, daily_w)
b = build_features(pm_x, wx_x, daily_x)
a = a[a["date"] == D].set_index("site_id")[FEATURES]
b = b[b["date"] == D].set_index("site_id")[FEATURES]
pd.testing.assert_frame_equal(a, b)
print(f"\nLeakage check PASSED: {len(a)} stations x {len(FEATURES)} features unchanged "
      "after scrambling everything from 16:00 on", D.date())
