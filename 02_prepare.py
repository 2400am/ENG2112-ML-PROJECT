"""
clean the hourly PM2.5, drop stations without enough data, and build the daily table.
writes  data/processed/sites.csv            stations actually used from here on
        data/processed/pm25_hourly_qc.pkl   cleaned hourly readings
        data/processed/pm25_daily.pkl       one row per station per day (the modelling table)
"""

import numpy as np
import pandas as pd
import config as c

# loading and sorting
pm = pd.read_csv(c.RAW / "pm25_hourly.csv", parse_dates=["time"])
pm = pm.sort_values(["site_id", "time"]).reset_index(drop=True)

# range check: below -10 or above 1500 = instrument fault
n_before = pm["pm25"].notna().sum()
bad = (pm["pm25"] < -10) | (pm["pm25"] > c.MAX_PLAUSIBLE_HOURLY)
pm.loc[bad, "pm25"] = np.nan
print("range check removed", n_before - pm["pm25"].notna().sum(), "values")

# stuck sensor: same value STUCK_HOURS+ in a row
changed = (pm["pm25"] != pm["pm25"].shift()) | (pm["site_id"] != pm["site_id"].shift())
run_id = changed.cumsum()
run_len = pm.groupby(run_id)["pm25"].transform("size")

stuck = (run_len >= c.STUCK_HOURS) & pm["pm25"].notna()
pm.loc[stuck, "pm25"] = np.nan
print("stuck-sensor check removed", stuck.sum(), "values")

# isolated spike: very high at ONE station only - kept, but flagged
sydney_median = pm.groupby("time")["pm25"].transform("median")
top1 = pm.groupby("time")["pm25"].transform("max")
top2 = pm["pm25"].where(pm["pm25"] < top1).groupby(pm["time"]).transform("max")
other_max = np.where(pm["pm25"] == top1, top2, top1)

pm["local_spike"] = (
    (pm["pm25"] > c.SPIKE_MIN)
    & (pm["pm25"] > c.SPIKE_RATIO * sydney_median.clip(lower=1))
    & (other_max < c.SPIKE_NEIGHBOUR_FRAC * pm["pm25"])
)
print(pm["local_spike"].sum(), "hours flagged as isolated spikes")

# daily means
pm["date"] = pm["time"].dt.normalize()
daily = pm.groupby(["site_id", "date"]).agg(
    pm25_mean=("pm25", "mean"),
    n_hours=("pm25", "count"),
    local_spike=("local_spike", "any"),
).reset_index()
print(f"\n{len(daily):,} station-days in total")

daily = daily[daily["n_hours"] >= c.MIN_HOURS_PER_DAY].drop(columns="n_hours")
print(f"{len(daily):,} valid station-days (at least {c.MIN_HOURS_PER_DAY} hours)")

# negative hourly noise was kept so it cancels out; a day that still averages
# below zero is set to 0 (needed later for log transforms).
print((daily["pm25_mean"] < 0).sum(), "daily means below zero set to 0")
daily["pm25_mean"] = daily["pm25_mean"].clip(lower=0)

# dropping stations with too little data
sites = pd.read_csv(c.RAW / "sites.csv").rename(columns={
    "Site_Id": "site_id", "SiteName": "site_name", "Latitude": "latitude",
    "Longitude": "longitude", "Region": "region"})

coverage = daily.groupby("site_id").agg(
    valid_days=("date", "size"), first_day=("date", "min"), last_day=("date", "max"))
coverage = sites.set_index("site_id")[["site_name"]].join(coverage)
coverage["valid_days"] = coverage["valid_days"].fillna(0).astype(int)
coverage["kept"] = coverage["valid_days"] >= c.MIN_VALID_DAYS
print(f"\nStation coverage (kept = at least {c.MIN_VALID_DAYS} valid days):")
print(coverage.sort_values("valid_days", ascending=False).to_string())

keep = coverage.index[coverage["kept"]]
sites = sites[sites["site_id"].isin(keep)]
daily = daily[daily["site_id"].isin(keep)]
pm = pm[pm["site_id"].isin(keep)]

# labels for later
year = daily["date"].dt.year.astype(str)
daily["split"] = np.select(
    [daily["date"] <= c.TRAIN_END, year == c.VAL_YEAR, year == c.TEST_YEAR],
    ["train", "val", "test"],
    default="other")
daily["fire_period"] = daily["date"].between(c.FIRE_START, c.FIRE_END)
daily["exceeds_who"] = daily["pm25_mean"] > c.WHO_24H

# saving the stuff
sites.to_csv(c.PROCESSED / "sites.csv", index=False)
pm.drop(columns="date").to_pickle(c.PROCESSED / "pm25_hourly_clean.pkl")
daily.to_pickle(c.PROCESSED / "pm25_daily.pkl")

# numbers for the report
by_split = daily.groupby("split").agg(
    stations=("site_id", "nunique"),
    station_days=("pm25_mean", "size"),
    who_exceed_rate=("exceeds_who", "mean"),
    nepm_exceed_rate=("pm25_mean", lambda x: (x > c.NEPM_24H).mean()))
by_fire = daily.groupby("fire_period").agg(
    station_days=("pm25_mean", "size"),
    who_exceed_rate=("exceeds_who", "mean"))

print(f"\n{len(sites)} stations kept, {len(daily):,} valid station-days, "
      f"{daily['local_spike'].sum()} days with a flagged spike")
print("\nBy split\n", by_split.round(3))
print("\nBy fire period\n", by_fire.round(3))

by_split.to_csv(c.OUTPUTS / "data_volume_by_split.csv")
by_fire.to_csv(c.OUTPUTS / "data_volume_by_fire_period.csv")
coverage.to_csv(c.OUTPUTS / "station_coverage.csv")

