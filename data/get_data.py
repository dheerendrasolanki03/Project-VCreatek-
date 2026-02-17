import requests # type: ignore
import pandas as pd # type: ignore
from datetime import datetime, timedelta
import time

# ======================================
# CONFIG
# ======================================

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIyWUNVMkYiLCJqdGkiOiI2OTkyYzQwNjBjNmFkNDY4MWNkNDgwMTEiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzcxMjI2MTE4LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzEyNzkyMDB9.AfDHeAZwIuTs6T92dN0LzJWiZRtMiHBQ_hw_wjB0GBo"
INSTRUMENT_FILE = "MCX.json"
OUTPUT_FILE = "crudeoil_1min_last300days.csv"

# ======================================
# LOAD INSTRUMENTS
# ======================================

inst_df = pd.read_json(INSTRUMENT_FILE)

inst_df["trading_symbol"] = inst_df["trading_symbol"].astype(str).str.upper()

# Filter CRUDEOIL FUTURES ONLY
crude_futures = inst_df[
    (inst_df["segment"] == "MCX_FO") &
    (inst_df["instrument_type"] == "FUT") &
    (inst_df["trading_symbol"].str.contains("CRUDEOIL FUT"))
]

print("Crude futures found:", len(crude_futures))

if crude_futures.empty:
    print("❌ No crude futures found.")
    exit()

# Select first contract (latest listed)
selected_contract = crude_futures.iloc[0]

instrument_key = selected_contract["instrument_key"]

print("Selected Contract:", selected_contract["trading_symbol"])
print("Instrument Key:", instrument_key)

# ======================================
# SAFE DATE RANGE (NO FUTURE DATE)
# ======================================

today = datetime.today()
end_date = today - timedelta(days=1)   # NEVER use today
start_date = end_date - timedelta(days=300)

print(f"Downloading from {start_date.date()} to {end_date.date()}")

# ======================================
# DOWNLOAD IN 30 DAY CHUNKS
# ======================================

all_data = []

current_start = start_date

while current_start < end_date:

    current_end = min(current_start + timedelta(days=30), end_date)

    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/1minute/{current_end.date()}/{current_start.date()}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    print(f"Downloading {current_start.date()} to {current_end.date()}")

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("❌ API Error:", response.text)
        break

    data = response.json()

    if "data" in data and "candles" in data["data"]:
        candles = data["data"]["candles"]
        all_data.extend(candles)

    time.sleep(0.5)  # prevent rate limit

    current_start = current_end + timedelta(days=1)

# ======================================
# CREATE DATAFRAME
# ======================================

if not all_data:
    print("❌ No data downloaded.")
    exit()

df = pd.DataFrame(all_data, columns=[
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest"
])

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

df.to_csv(OUTPUT_FILE, index=False)

print("✅ Data Saved Successfully")
print("Rows:", len(df))
