"""
project settings
"""

from pathlib import Path

# folders and paths
DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = Path(__file__).parent / "outputs"

for p in (RAW, PROCESSED, OUTPUTS):
    p.mkdir(parents= True, exist_ok= True)

# study region and dates
REGION = "Greater Sydney"
BBOX = dict(lat_min=-34.35, lat_max=-33.45, lon_min=150.55, lon_max=151.35)
STUDY_START, STUDY_END = "2016-01-01", "2024-12-31"
TRAIN_END = "2022-12-31"           # train: 2016-2022
VAL_YEAR = "2023"                  # tuning / threshold choice only
TEST_YEAR = "2024"

# hourly data cleaning 
MAX_PLAUSIBLE_HOURLY = 1500   # ug/m3; above = instrument fault (highest real reading is 1085)
STUCK_HOURS = 6 # same value 6+ hours in a row = stuck sensor
# isolated local spikes
SPIKE_MIN = 100              # rule 1: the reading must be above 100 ug/m3
SPIKE_RATIO = 10             # rule 2: and more than 10x the Sydney median that hour
SPIKE_NEIGHBOUR_FRAC = 0.5   # rule 3: and no other station above half its value (else it's a regional plume)

# daily table
MIN_HOURS_PER_DAY = 18     # a daily mean needs at least 18 of 24 hours (75% completeness rule)
MIN_VALID_DAYS = 365       # drop stations with less than a year of valid days in total

# targets
WHO_24H = 15.0             # WHO 2021 24-hour PM2.5 guideline (ug/m3) - the main threshold
NEPM_24H = 25.0            # Australian national standard - reported as a secondary threshold

# black summer bush fire smoke period
FIRE_START, FIRE_END = "2019-10-01", "2020-02-29"

# forecasting
# forecast issued at 16:00 on day D using only data before then; target = day D+1.
FORECAST_CUTOFF_HOUR = 16

# feature groups
OWN = ["pm_today", "pm_today_max", "pm_last3h", "pm_yesterday", "pm_3day", "pm_7day"]
NETWORK = ["net_today", "net_yesterday"]
WEATHER = ["wx_temp", "wx_rh", "wx_wind", "wx_wind_u", "wx_wind_v", "wx_blh", "wx_rain"]
CALENDAR = ["doy_sin", "doy_cos", "weekend_tomorrow"]
FEATURES = OWN + NETWORK + WEATHER + CALENDAR

# evaluation
SEED = 42                # fixes all randomness, so results are identical every run
N_BOOT = 1000            # resamples for 95% confidence intervals
MIN_PRAUC_GAIN = 0.05    # pre-registered success: GBM beats the best baseline's PR-AUC by >= 0.05, CI above 0