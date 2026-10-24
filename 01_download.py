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

def get_sites():
    #download stations
    r = requests.get(f"{API}/get_SiteDetails", timeout= 60)
    sites = pd.DataFrame(r.json())
    sydney = sites[
        sites["Longitude"].between(c.BBOX["lon_min"], c.BBOX["lon_max"]) 
        & sites["Latitude"].between(c.BBOX["lat_min"], c.BBOX["lat_max"]) 
        & (sites["Region"] != "Test Region")
    ]
    return sydney

sydney = get_sites()
print(sydney)
print(len(sydney))