from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import numpy as np
import csv
from datetime import datetime

# -------------------------
# Define Base Directory First
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Point Flask to custom template folder
# -------------------------
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "web", "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# -------------------------
# Model file paths
# -------------------------
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Load model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Transaction type encoding
transaction_mapping = {
    "PAYMENT": 1,
    "TRANSFER": 2,
    "CASH_OUT": 3,
    "DEBIT": 4,
    "CASH_IN": 5
}


# ---------------------------------------
#                  ROUTES
# ---------------------------------------
@app.route("/")
def home():
    return render_template("predict_form.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1️⃣ Read input values
        amount = float(request.form["amount"])
        tx_type = request.form["transaction_type"]
        old_balance = float(request.form["old_balance"])
        new_balance = float(request.form["new_balance"])

        # 2️⃣ Encode transaction type
        tx_value = transaction_mapping.get(tx_type, 0)

        # 3️⃣ Prepare input
        row = np.array([[amount, tx_value, old_balance, new_balance]])
        row_scaled = scaler.transform(row)

        # 4️⃣ Predict fraud probability
        probability = round(model.predict_proba(row_scaled)[0][1], 2)

        # 5️⃣ Apply custom threshold so model detects more fraud cases
        THRESHOLD = 0.30  # lower threshold makes model more sensitive
        prediction = 1 if probability >= THRESHOLD else 0

        # 6️⃣ Result message
        result = "🚨 FRAUD DETECTED!" if prediction == 1 else "✅ SAFE TRANSACTION"
        color = "red" if prediction == 1 else "green"

        # 7️⃣ Log prediction to CSV with UTF-8 support
        log_file = os.path.join(BASE_DIR, "..", "predictions_log.csv")
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "amount", "type", "old_balance", "new_balance", "prediction", "confidence"])

            writer.writerow([datetime.now(), amount, tx_type, old_balance, new_balance, result, probability])

        # 8️⃣ Send result to UI
        return render_template(
            "predict_form.html",
            result=result,
            confidence=probability,
            color=color
        )

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ---------------------------------------
#                 RUN APP
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
