import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
import matplotlib.pyplot as plt # type: ignore

model = joblib.load(r"C:\Users\Solanki\OneDrive\Desktop\i1\crude_oil1\models\lightgbm_model.pkl")

df = pd.read_csv("C:\\Users\\Solanki\\OneDrive\\Desktop\\i1\\crude_oil1\\data\\crudeoil_1min_last300days.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df[df["volume"] > 0].reset_index(drop=True)

LOOKAHEAD = 10

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_20"] = df["close"].rolling(20).mean()
df["ma_ratio"] = df["ma_5"] / df["ma_20"]
df["volatility_20"] = df["log_return"].rolling(20).std()
df["vol_change"] = df["volume"].pct_change()
df["oi_change"] = df["open_interest"].pct_change()
df["hour"] = df["timestamp"].dt.hour

df = df.dropna().reset_index(drop=True)

features = [
    "log_return",
    "ma_5",
    "ma_20",
    "ma_ratio",
    "volatility_20",
    "volume",
    "vol_change",
    "open_interest",
    "oi_change",
    "hour"
]

X = df[features]

probs = model.predict_proba(X)[:, 1]
df["prob"] = probs

initial_capital = 100000
capital = initial_capital

transaction_cost = 0.0002
slippage = 0.0001

long_threshold = 0.60
short_threshold = 0.40

position = 0
entry_price = 0

equity_curve = []
trade_returns = []

for i in range(len(df) - LOOKAHEAD):

    price = df.loc[i, "close"]
    prob = df.loc[i, "prob"]

    if position == 0:

        if prob > long_threshold:
            position = 1
            entry_price = price * (1 + slippage)
            capital -= capital * transaction_cost

        elif prob < short_threshold:
            position = -1
            entry_price = price * (1 - slippage)
            capital -= capital * transaction_cost

    else:

        exit_price = df.loc[i + LOOKAHEAD, "close"]

        if position == 1:
            exit_price *= (1 - slippage)
            trade_return = (exit_price - entry_price) / entry_price

        elif position == -1:
            exit_price *= (1 + slippage)
            trade_return = (entry_price - exit_price) / entry_price

        capital *= (1 + trade_return)
        capital -= capital * transaction_cost

        trade_returns.append(trade_return)
        position = 0

    equity_curve.append(capital)

equity_curve = np.array(equity_curve)
trade_returns = np.array(trade_returns)

total_return = (capital - initial_capital) / initial_capital * 100

if len(trade_returns) > 1 and np.std(trade_returns) != 0:
    sharpe_ratio = (
        np.mean(trade_returns) / np.std(trade_returns)
    ) * np.sqrt(252 * 390)
else:
    sharpe_ratio = 0

rolling_max = np.maximum.accumulate(equity_curve)
drawdown = (equity_curve - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

print("\nBACKTEST RESULTS")
print("Final Capital:", round(capital, 2))
print("Total Return:", round(total_return, 2), "%")
print("Total Trades:", len(trade_returns))
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Max Drawdown:", round(max_drawdown, 2), "%")

plt.figure(figsize=(12,6))
plt.plot(equity_curve)
plt.title("Equity Curve")
plt.show()
