import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
CSV_FILE = "stock_data.csv"

app = Flask(__name__, template_folder="templates")


def load_close_series():
    df = pd.read_csv(CSV_FILE)
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found in {CSV_FILE}")
    close = pd.to_numeric(df["Close"], errors="coerce").dropna().astype(float).values.reshape(-1, 1)
    return close  # (n,1)


def inverse_scale(scaler, x_scaled):
    arr = np.array(x_scaled).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)


# Load artifacts
lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "r") as f:
    metrics = json.load(f)

TIME_STEP = int(metrics.get("time_step", 60))

# Load data (once at start)
close = load_close_series()
close_scaled = scaler.transform(close)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(metrics)


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Returns last N days actual close prices for plotting.
    Query param: n (default 60)
    """
    n = int(request.args.get("n", 60))
    n = max(10, min(n, len(close)))

    hist = close[-n:, 0].astype(float).tolist()
    labels = [f"Day {i+1}" for i in range(n)]  # simple labels (works even without Date column)

    return jsonify({
        "n": n,
        "labels": labels,
        "close": hist,
        "last_close": float(close[-1, 0])
    })


@app.route("/api/predict_next", methods=["POST"])
def api_predict_next():
    """
    body: {"model":"lstm"} or {"model":"lr"}
    """
    data = request.get_json(silent=True) or {}
    model_name = (data.get("model") or "lstm").lower()

    last_close = float(close[-1, 0])

    if model_name == "lr":
        idx_next = len(close_scaled)
        pred_scaled = float(lr_model.predict(np.array([[idx_next]])).reshape(-1)[0])
        pred_price = float(inverse_scale(scaler, [pred_scaled])[0])
    else:
        if len(close_scaled) < TIME_STEP:
            return jsonify({"error": f"Not enough data for time_step={TIME_STEP}"}), 400

        seq = close_scaled[-TIME_STEP:, 0].reshape(1, TIME_STEP, 1)
        pred_scaled = float(lstm_model.predict(seq, verbose=0).reshape(-1)[0])
        pred_price = float(inverse_scale(scaler, [pred_scaled])[0])

    change = pred_price - last_close
    pct = (change / last_close) * 100 if last_close != 0 else 0.0

    return jsonify({
        "model": model_name,
        "last_close": last_close,
        "predicted_next_close": pred_price,
        "change": float(change),
        "change_percent": float(pct)
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
