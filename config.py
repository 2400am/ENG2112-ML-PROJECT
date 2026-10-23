from pathlib import Path

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

