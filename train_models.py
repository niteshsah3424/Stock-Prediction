import os
import json
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


df = pd.read_csv("stock_data.csv")

if "Close" not in df.columns:
    raise ValueError(f"Close column not found. Columns are: {list(df.columns)}")

df = df[["Close"]].copy()
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(inplace=True)

close = df["Close"].values.reshape(-1, 1)

TIME_STEP = 60
if len(close) <= TIME_STEP + 10:
    raise ValueError(f"Not enough rows. Need > {TIME_STEP+10}, got {len(close)}")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

# -----------------------------
# Linear Regression
# -----------------------------
X = np.arange(len(scaled)).reshape(-1, 1)
y = scaled.reshape(-1)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_rmse_scaled = math.sqrt(mean_squared_error(y_test, lr_pred))

y_test_price = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
lr_pred_price = scaler.inverse_transform(lr_pred.reshape(-1, 1)).reshape(-1)
lr_rmse_price = math.sqrt(mean_squared_error(y_test_price, lr_pred_price))

print("✅ LR RMSE (scaled):", lr_rmse_scaled)
print("✅ LR RMSE (price):", lr_rmse_price)

# -----------------------------
# LSTM
# -----------------------------
X_lstm, y_lstm = create_dataset(scaled, TIME_STEP)
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

split_lstm = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]
y_train_lstm, y_test_lstm = y_lstm[:split_lstm], y_lstm[split_lstm:]

lstm_model = Sequential([
    Input(shape=(TIME_STEP, 1)),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=8, batch_size=32, verbose=1)

lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).reshape(-1)
lstm_rmse_scaled = math.sqrt(mean_squared_error(y_test_lstm, lstm_pred))

y_test_lstm_price = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).reshape(-1)
lstm_pred_price = scaler.inverse_transform(lstm_pred.reshape(-1, 1)).reshape(-1)
lstm_rmse_price = math.sqrt(mean_squared_error(y_test_lstm_price, lstm_pred_price))

print("✅ LSTM RMSE (scaled):", lstm_rmse_scaled)
print("✅ LSTM RMSE (price):", lstm_rmse_price)

# -----------------------------
# Save models + metrics
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(lr, "models/lr_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
lstm_model.save("models/lstm_model.keras")

metrics = {
    "time_step": TIME_STEP,
    "total_rows": int(len(close)),
    "lr_rmse_scaled": float(lr_rmse_scaled),
    "lr_rmse_price": float(lr_rmse_price),
    "lstm_rmse_scaled": float(lstm_rmse_scaled),
    "lstm_rmse_price": float(lstm_rmse_price),
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Saved models in models/ folder")
