import pandas as pd
import os

# ===== CONFIGURE HORIZON HERE =====
HORIZON = 1  # 1 = 100ms, 5 = 500ms, 10 = 1s
# ==================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

input_file = os.path.join(DATA_DIR, "raw_orderbook.csv")
output_file = os.path.join(DATA_DIR, f"hft_dataset_binary_{HORIZON*100}ms.csv")

data = pd.read_csv(input_file)

# ===== Feature Engineering =====

data["mid_price"] = (data["best_bid_price"] + data["best_ask_price"]) / 2
data["spread"] = data["best_ask_price"] - data["best_bid_price"]

data["obi"] = (data["bid_vol_sum"] - data["ask_vol_sum"]) / (
              data["bid_vol_sum"] + data["ask_vol_sum"])

data["micro_price"] = (
    (data["best_ask_price"] * data["best_bid_qty"] +
     data["best_bid_price"] * data["best_ask_qty"]) /
    (data["best_bid_qty"] + data["best_ask_qty"])
)

# ===== Label Building =====

data["future_mid"] = data["mid_price"].shift(-HORIZON)

def label_binary(row):
    if row["future_mid"] > row["mid_price"]:
        return 1   # UP
    elif row["future_mid"] < row["mid_price"]:
        return 0   # DOWN
    else:
        return None

data["label"] = data.apply(label_binary, axis=1)
data = data.dropna()

# ===== Save Versioned Dataset =====

data.to_csv(output_file, index=False)

print(f"Dataset built for horizon = {HORIZON*100} ms")
print("Shape:", data.shape)
print("Saved to:", output_file)
