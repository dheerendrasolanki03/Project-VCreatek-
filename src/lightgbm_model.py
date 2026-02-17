import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
from lightgbm import LGBMClassifier # type: ignore
from sklearn.metrics import (confusion_matrix,classification_report,accuracy_score,roc_auc_score) # type: ignore

df = pd.read_csv(r"C:\Users\Solanki\OneDrive\Desktop\i1\crude_oil1\data\crudeoil_1min_last300days.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

df = df[df["volume"] > 0].reset_index(drop=True)

print("Total usable rows:", len(df))

LOOKAHEAD = 10

df["future_close"] = df["close"].shift(-LOOKAHEAD)
df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

threshold = 0.001

df["signal"] = np.where(df["future_return"] > threshold, 1, 0)

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
y = df["signal"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=64,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    verbosity=-1,  
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\n CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\n ACCURACY:", accuracy_score(y_test, y_pred))

print(" ROC-AUC:", roc_auc_score(y_test, y_prob))

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n FEATURE IMPORTANCE:")
print(importance)

latest_features = X.iloc[-1:].values
prediction = model.predict(latest_features)[0]
probability = model.predict_proba(latest_features)[0]

print("\n")
if prediction == 1:
    print(" MODEL SIGNAL: BUY")
    print("Confidence:", round(probability[1]*100, 2), "%")
else:
    print(" MODEL SIGNAL: SELL")
    print("Confidence:", round(probability[0]*100, 2), "%")

joblib.dump(model, "lightgbm_model.pkl")
print("\nModel saved as 'lightgbm_model.pkl'")
