import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import joblib # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore

df = pd.read_csv(r"C:\Users\Solanki\OneDrive\Desktop\i1\crude_oil1\data\crudeoil_1min_last300days.csv")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)

future_return = df['close'].shift(-1) - df['close']
df['target'] = np.where(future_return > 0, 1, 0)

df['return'] = df['close'].pct_change()
df['ma_5'] = df['close'].rolling(5).mean()
df['ma_10'] = df['close'].rolling(10).mean()
df['volatility'] = df['return'].rolling(10).std()

df.dropna(inplace=True)

features = ['open', 'high', 'low', 'close', 'volume',
            'return', 'ma_5', 'ma_10', 'volatility']

X = df[features]
y = df['target']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 20
X_seq, y_seq = create_sequences(X_scaled, y.values, TIME_STEPS)

split = int(0.8 * len(X_seq))
X_test = X_seq[split:]
y_test = y_seq[split:]

df_test = df.iloc[TIME_STEPS + split:].copy()

model = load_model("C:\\Users\\Solanki\\OneDrive\\Desktop\\i1\\crude_oil1\\models\\lstm_model.h5")

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

df_test['signal'] = np.where(y_pred == 1, 1, -1)

df_test['market_return'] = df_test['close'].pct_change()

df_test['strategy_return'] = df_test['signal'].shift(1) * df_test['market_return']

df_test.dropna(inplace=True)

df_test['cumulative_strategy'] = (1 + df_test['strategy_return']).cumprod()
df_test['cumulative_market'] = (1 + df_test['market_return']).cumprod()

total_return = df_test['cumulative_strategy'].iloc[-1] - 1
market_return = df_test['cumulative_market'].iloc[-1] - 1

print("\nStrategy Return:", round(total_return * 100, 2), "%")
print("Buy & Hold Return:", round(market_return * 100, 2), "%")

# Strategy Equity Curve
plt.figure()
plt.plot(df_test['cumulative_strategy'])
plt.title("LSTM Strategy Cumulative Return")
plt.show()

# Buy & Hold
plt.figure()
plt.plot(df_test['cumulative_market'])
plt.title("Buy & Hold Return")
plt.show()

# Strategy vs Market
plt.figure()
plt.plot(df_test['cumulative_strategy'])
plt.plot(df_test['cumulative_market'])
plt.legend(["Strategy", "Buy & Hold"])
plt.title("LSTM Strategy vs Buy & Hold")
plt.show()

# Drawdown
rolling_max = df_test['cumulative_strategy'].cummax()
drawdown = df_test['cumulative_strategy'] / rolling_max - 1

plt.figure()
plt.plot(drawdown)
plt.title("LSTM Strategy Drawdown")
plt.show()
