import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# ===== CONFIGURE HORIZON HERE =====
HORIZON = 1 # Must match dataset horizon
# ==================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

data_file = os.path.join(DATA_DIR, f"hft_dataset_binary_{HORIZON*100}ms.csv")
data = pd.read_csv(data_file)

X = data[["spread", "obi", "micro_price"]]
y = data["label"]

# Time-aware split
split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ===== Save Versioned Model =====
model_path = os.path.join(BASE_DIR, f"xgb_model_{HORIZON*100}ms.json")
model.save_model(model_path)

print("\nModel saved to:", model_path)
