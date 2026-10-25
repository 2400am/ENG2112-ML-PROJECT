"""
clean the hourly PM2.5, drop stations without enough data, and build the daily table.
writes  data/processed/sites.csv            stations actually used from here on
        data/processed/pm25_hourly_qc.pkl   cleaned hourly readings
        data/processed/pm25_daily.pkl       one row per station per day (the modelling table)
"""

import numpy as np
import pandas as pd
import config as c

pm = pd.read_csv(c.RAW / "pm25_hourly.csv", parse_dates=["time"])
high = pm[pm["pm25"] > 500]
print(len(high), "readings above 500")
print(high.sort_values("pm25", ascending=False).head(10))