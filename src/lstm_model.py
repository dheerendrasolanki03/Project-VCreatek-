import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import joblib # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # type: ignore

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

df = pd.read_csv(r"C:\Users\Solanki\OneDrive\Desktop\i1\crude_oil1\data\crudeoil_1min_last300days.csv")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)

future_return = df['close'].shift(-1) - df['close']

df['target'] = np.where(future_return > 0, 1, 0)  # 1=BUY, 0=SELL

df.dropna(inplace=True)

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

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

latest_sequence = X_seq[-1].reshape(1, TIME_STEPS, len(features))
latest_prob = model.predict(latest_sequence)[0][0]

if latest_prob > 0.5:
    print("\n LSTM SIGNAL: BUY")
else:
    print("\n LSTM SIGNAL: SELL")

joblib.save(model, "lstm_model.h5")
print("\nModel saved as 'lstm_model.h5'")