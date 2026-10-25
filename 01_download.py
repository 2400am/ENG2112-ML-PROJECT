"""
downloads the raw data for the project:
1. sites.csv          - list of NSW government air-quality monitoring stations in the Sydney box
2. pm25_hourly.csv    - hourly PM2.5 measured at those stations, 2016-2024 (NSW DCCEEW Air Quality API)
3. weather_hourly.csv - hourly weather at each station's location (ERA5 reanalysis via Open-Meteo)

API guide: https://www.environment.nsw.gov.au/sites/default/files/air-quality-application-programming-interface-user-guide-210346.pdf
"""

import time
import requests
import pandas as pd
import config as c

API = "https://data.airquality.nsw.gov.au/api/Data"
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"
CHUNKS = c.RAW / "chunks"
CHUNKS.mkdir(exist_ok=True)

def get_sites():
    """download station list and keep the stations in sydney box"""
    r = requests.get(f"{API}/get_SiteDetails", timeout= 60)
    sites = pd.DataFrame(r.json())
    sydney = sites[
        sites["Longitude"].between(c.BBOX["lon_min"], c.BBOX["lon_max"]) 
        & sites["Latitude"].between(c.BBOX["lat_min"], c.BBOX["lat_max"]) 
        & (sites["Region"] != "Test Region")
    ]
    return sydney

def download(site_ids, start, end):
    """download hourly pm2.5 for these stations, start included, end not included"""
    body = {
        "Parameters": ["PM2.5"],
        "Sites": site_ids,
        "StartDate": start,
        "EndDate": end,
        "Categories": ["Averages"],
        "Subcategories": ["Hourly"],
        "Frequency": ["Hourly average"],
    }
    for attempt in range(3):
        try:
            r = requests.post(f"{API}/get_Observations", json=body, timeout=120)
            if r.status_code == 200:
                return pd.DataFrame(r.json())
            print("server error", r.status_code, "- retrying")
        except requests.RequestException as e:
            print("connection problem:", type(e).__name__, "- retrying")
        time.sleep(10)
    raise RuntimeError(f"Gave up on {start} for stations {site_ids}")

def tidy(raw):
    """keep site_id, a proper hour start timestamp, and the pm2.5 value"""
    hour_start = raw["Hour"].astype(int) - 1
    return pd.DataFrame({
        "site_id": raw["Site_Id"],
        "time": pd.to_datetime(raw["Date"]) + pd.to_timedelta(hour_start, unit="h"),
        "pm25": pd.to_numeric(raw["Value"], errors="coerce")
    })

def get_weather(lat, lon):
    """Hourly weather at one location for the whole study period (ERA5 via Open-Meteo)."""
    p = {
        "latitude": lat,
        "longitude": lon,
        "start_date": c.STUDY_START,
        "end_date": c.STUDY_END,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,boundary_layer_height",
        "timezone": "Australia/Brisbane",

    }
    for attempt in range(6):
        try:
            r = requests.get(WEATHER_API, params=p, timeout=300)
            if r.status_code == 200:
                return pd.DataFrame(r.json()["hourly"])
            print("weather error", r.status_code, "- retrying")
        except requests.RequestException as e:
            print("connection problem:", type(e).__name__, "- retrying")
        time.sleep(60 * (attempt + 1))
    raise RuntimeError(f"gave up on weather at {lat}, {lon}")


# sydney stations
sydney = get_sites()

# which stations measure pm2.5
test = download(sydney["Site_Id"].tolist(), "2024-01-01", "2024-01-02")

# keep them in groups of 5 and save
pm_ids = sorted(test["Site_Id"].unique().tolist())
groups = [pm_ids[i:i+5] for i in range(0, len(pm_ids), 5)]
sydney = sydney[sydney["Site_Id"].isin(pm_ids)]
sydney.to_csv(c.RAW / "sites.csv", index=False)
print(len(pm_ids), "stations measure PM2.5")
print(groups)

# download every month (skips months alr saved)
months = pd.date_range(c.STUDY_START, c.STUDY_END, freq="MS")
for m in months:
    end = m + pd.offsets.MonthBegin(1)
    start_str = m.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    path = CHUNKS / f"pm25_{m:%Y-%m}.csv"
    if path.exists():
        continue

    raw = pd.concat([download(g, start_str, end_str) for g in groups], ignore_index=True)
    month = tidy(raw)
    month.to_csv(path, index=False)
    print(f"{m:%Y-%m}: {len(month):,} rows")

# combine every month into one file
files = sorted(CHUNKS.glob("pm25_*.csv"))
pm = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
pm.to_csv(c.RAW / "pm25_hourly.csv", index=False)
print(f"pm25_hourly.csv: {len(pm):,} rows, {pm['site_id'].nunique()} stations, "
      f"{pm['pm25'].notna().sum():,} with a value")

# weather for each station
for _, s in sydney.iterrows():
    path = CHUNKS / f"weather_{s['Site_Id']}.csv"
    if path.exists():
        continue

    w = get_weather(s["Latitude"], s["Longitude"])
    w["site_id"] = s["Site_Id"]
    w.to_csv(path, index=False)
    print(f"{s['SiteName']}: {len(w):,} hours")
    time.sleep(2)

# combine every station into one file
files = sorted(CHUNKS.glob("weather_*.csv"))
wx = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
wx.to_csv(c.RAW / "weather_hourly.csv", index=False)
print(f"weather_hourly.csv: {len(wx):,} rows, {wx['site_id'].nunique()} stations")