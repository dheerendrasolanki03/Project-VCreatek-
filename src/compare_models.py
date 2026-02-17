import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import joblib # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore

df = pd.read_csv("C:\\Users\\Solanki\\OneDrive\\Desktop\\i1\\crude_oil1\\data\\crudeoil_1min_last300days.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)

df_lgb = df.copy()

df_lgb["log_return"] = np.log(df_lgb["close"] / df_lgb["close"].shift(1))
df_lgb["ma_5"] = df_lgb["close"].rolling(5).mean()
df_lgb["ma_20"] = df_lgb["close"].rolling(20).mean()
df_lgb["ma_ratio"] = df_lgb["ma_5"] / df_lgb["ma_20"]
df_lgb["volatility_20"] = df_lgb["log_return"].rolling(20).std()
df_lgb["vol_change"] = df_lgb["volume"].pct_change()
df_lgb["oi_change"] = df_lgb["open_interest"].pct_change()
df_lgb["hour"] = df_lgb.index.hour

lgb_features = [
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

df_lgb.dropna(inplace=True)

X_lgb = df_lgb[lgb_features]

df_lstm = df.copy()

df_lstm["return"] = df_lstm["close"].pct_change()
df_lstm["ma_5"] = df_lstm["close"].rolling(5).mean()
df_lstm["ma_10"] = df_lstm["close"].rolling(10).mean()
df_lstm["volatility"] = df_lstm["return"].rolling(10).std()

lstm_features = [
    "open", "high", "low", "close", "volume",
    "return", "ma_5", "ma_10", "volatility"
]

df_lstm.dropna(inplace=True)

X_lstm = df_lstm[lstm_features]

scaler = MinMaxScaler()
X_lstm_scaled = scaler.fit_transform(X_lstm)

def create_sequences(X, time_steps=20):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

TIME_STEPS = 20
X_seq = create_sequences(X_lstm_scaled, TIME_STEPS)

split_lgb = int(0.8 * len(X_lgb))
split_lstm = int(0.8 * len(X_seq))

X_lgb_test = X_lgb.iloc[split_lgb:]
X_lstm_test = X_seq[split_lstm:]

test_index = df_lgb.index[split_lgb:]
df_test = df_lgb.iloc[split_lgb:].copy()

lgb_model = joblib.load("C:\\Users\\Solanki\\OneDrive\\Desktop\\i1\\crude_oil1\\models\\lightgbm_model.pkl")
lstm_model = load_model("C:\\Users\\Solanki\\OneDrive\\Desktop\\i1\\crude_oil1\\models\\lstm_model.h5")

# LightGBM
lgb_preds = lgb_model.predict(X_lgb_test)
df_test["lgb_signal"] = np.where(lgb_preds == 1, 1, -1)

# LSTM
lstm_probs = lstm_model.predict(X_lstm_test)
lstm_preds = (lstm_probs > 0.5).astype(int).flatten()

lstm_aligned = lstm_preds[-len(df_test):]
df_test["lstm_signal"] = np.where(lstm_aligned == 1, 1, -1)

def backtest(data, signal_column, cost=0.0001):
    data = data.copy()

    data["market_return"] = data["close"].pct_change()

    data["position"] = data[signal_column].shift(1)

    data["trade"] = data["position"].diff().abs()
    data["cost"] = data["trade"] * cost

    data["strategy_return"] = data["position"] * data["market_return"] - data["cost"]

    data.dropna(inplace=True)

    data["equity"] = (1 + data["strategy_return"]).cumprod()

    total_return = data["equity"].iloc[-1] - 1
    sharpe = np.sqrt(252*390) * data["strategy_return"].mean() / data["strategy_return"].std()

    rolling_max = data["equity"].cummax()
    drawdown = data["equity"] / rolling_max - 1
    max_dd = drawdown.min()

    return data, total_return, sharpe, max_dd

lgb_bt, lgb_ret, lgb_sharpe, lgb_dd = backtest(df_test, "lgb_signal")
lstm_bt, lstm_ret, lstm_sharpe, lstm_dd = backtest(df_test, "lstm_signal")

df_test["market_return"] = df_test["close"].pct_change()
df_test["buy_hold"] = (1 + df_test["market_return"]).cumprod()

print("\n===== LightGBM =====")
print("Return:", round(lgb_ret*100,2), "%")
print("Sharpe:", round(lgb_sharpe,2))
print("Max DD:", round(lgb_dd*100,2), "%")

print("\n===== LSTM =====")
print("Return:", round(lstm_ret*100,2), "%")
print("Sharpe:", round(lstm_sharpe,2))
print("Max DD:", round(lstm_dd*100,2), "%")

plt.figure()
plt.plot(lgb_bt["equity"])
plt.plot(lstm_bt["equity"])
plt.plot(df_test["buy_hold"])
plt.legend(["LightGBM", "LSTM", "Buy & Hold"])
plt.title("Equity Curve Comparison")
plt.show()
