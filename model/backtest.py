import pandas as pd
import os
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ===== CONFIGURE HORIZON HERE =====
HORIZON = 1
# ==================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Load versioned dataset
data = pd.read_csv(os.path.join(DATA_DIR, f"hft_dataset_binary_{HORIZON*100}ms.csv"))

# Load raw orderbook
raw = pd.read_csv(os.path.join(DATA_DIR, "raw_orderbook.csv"))

# Align lengths
raw = raw.iloc[:len(data)].reset_index(drop=True)
data = data.reset_index(drop=True)

# Features
X = data[["spread", "obi", "micro_price"]]

# Load trained model
model = XGBClassifier()
model.load_model(os.path.join(BASE_DIR, f"xgb_model_{HORIZON*100}ms.json"))

pred = model.predict(X)

# Prices
mid_price = (raw["best_bid_price"] + raw["best_ask_price"]) / 2
future_mid = mid_price.shift(-HORIZON)

entry_ask = raw["best_ask_price"]
entry_bid = raw["best_bid_price"]

# ===== Backtest =====

pnl = []

for i in range(len(pred)-HORIZON):
    if pred[i] == 1:   # UP → Buy
        trade_pnl = future_mid[i] - entry_ask[i]
    else:              # DOWN → Sell
        trade_pnl = entry_bid[i] - future_mid[i]
    pnl.append(trade_pnl)

pnl = np.array(pnl)
pnl = pnl[~np.isnan(pnl)]

equity = np.cumsum(pnl)

# Sharpe Ratio
sharpe = 0
if pnl.std() != 0:
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(len(pnl))

print("\n===== BACKTEST RESULT =====")
print("Horizon:", HORIZON*100, "ms")
print("Total Trades:", len(pnl))
print("Total PnL:", round(equity[-1], 6))
print("Average PnL per Trade:", round(pnl.mean(), 8))
print("Sharpe Ratio:", round(sharpe, 3))

# ===== Save Results to Log File =====
log_file = os.path.join(BASE_DIR, "backtest_summary.csv")

if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("Horizon_ms,TotalTrades,TotalPnL,Sharpe\n")

with open(log_file, "a") as f:
    f.write(f"{HORIZON*100},{len(pnl)},{equity[-1]},{sharpe}\n")

print("Results logged in backtest_summary.csv")

# ===== Plot Equity Curve =====
plt.figure()
plt.plot(equity)
plt.title(f"Equity Curve - Horizon {HORIZON*100}ms")
plt.xlabel("Trade Number")
plt.ylabel("Cumulative PnL")
plt.show()
