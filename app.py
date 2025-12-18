from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

# ------------------ FIXED PATHS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

LOG_FILE = os.path.join(BASE_DIR, "predictions_log.csv")   # ✔ SAME FILE FOR DASHBOARD + PREDICTIONS

TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "web", "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load ML model
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

transaction_mapping = {
    "PAYMENT": 1, 
    "TRANSFER": 2, 
    "CASH_OUT": 3, 
    "DEBIT": 4, 
    "CASH_IN": 5
}


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        amount = float(request.form["amount"])
        tx_type = transaction_mapping.get(request.form["transaction_type"], 0)
        old_balance = float(request.form["old_balance"])
        new_balance = float(request.form["new_balance"])

        row = np.array([[amount, tx_type, old_balance, new_balance]])
        row_scaled = scaler.transform(row)

        probability = round(model.predict_proba(row_scaled)[0][1], 2)
        result = "🚨 FRAUD DETECTED!" if probability >= 0.30 else "✅ SAFE TRANSACTION"

        # Save to log
        new_data = pd.DataFrame([[amount, tx_type, old_balance, new_balance, probability, result]],
                                columns=["amount", "type", "old_balance", "new_balance", "confidence", "prediction"])

        if os.path.exists(LOG_FILE):
            new_data.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            new_data.to_csv(LOG_FILE, index=False)

        return render_template("results.html", prediction=result, score=probability)

    return render_template("predict_form.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():

    # ❗ USE SAME LOG PATH
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        return render_template("dashboard.html", no_data=True)

    df = pd.read_csv(LOG_FILE)

    df["is_fraud"] = df["prediction"].apply(lambda x: 1 if "FRAUD" in str(x).upper() else 0)

    total = len(df)
    fraud = df["is_fraud"].sum()
    safe = total - fraud
    rate = round((fraud / total) * 100, 2)

    charts = {}

    # Helper to convert plt → base64
    def generate_chart(fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()
        return encoded

    # Pie chart
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([safe, fraud], labels=["Safe", "Fraud"], autopct="%1.1f%%",
           colors=["#28a745", "#dc3545"])
    ax.set_title("Fraud vs Safe Distribution")
    charts["pie"] = generate_chart(fig)
    plt.close(fig)

    # Bar chart by type
    fig, ax = plt.subplots(figsize=(5, 4))
    df.groupby("type")["is_fraud"].sum().plot(kind="bar", color="#ff9800", ax=ax)
    ax.set_title("Fraud Count by Transaction Type")
    charts["bar"] = generate_chart(fig)
    plt.close(fig)

    # Line plot
    fig, ax = plt.subplots(figsize=(5, 4))
    df["confidence"].plot(kind="line", marker="o", color="#6f42c1", ax=ax)
    ax.set_title("Model Confidence Trend")
    charts["line"] = generate_chart(fig)
    plt.close(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    df["amount"].plot(kind="hist", bins=10, color="#007bff", ax=ax)
    ax.set_title("Transaction Amount Distribution")
    charts["hist"] = generate_chart(fig)
    plt.close(fig)

    return render_template(
        "dashboard.html",
        no_data=False,
        total=total,
        fraud=fraud,
        safe=safe,
        rate=rate,
        charts=charts
    )


if __name__ == "__main__":
    app.run(debug=True)
