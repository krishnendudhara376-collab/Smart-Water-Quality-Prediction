import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from database import init_db, save_prediction, get_all_predictions

app = Flask(__name__)
init_db()

# Load trained artifacts
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/metrics.json", "r") as f:
    metadata = json.load(f)

def prepare_input(form_data):
    raw = {
        'ph': float(form_data['ph']),
        'Hardness': float(form_data['Hardness']),
        'Solids': float(form_data['Solids']),
        'Chloramines': float(form_data['Chloramines']),
        'Sulfate': float(form_data['Sulfate']),
        'Conductivity': float(form_data['Conductivity']),
        'Organic_carbon': float(form_data['Organic_carbon']),
        'Trihalomethanes': float(form_data['Trihalomethanes']),
        'Turbidity': float(form_data['Turbidity'])
    }
    df_in = pd.DataFrame([raw])
    # Same Feature Engineering as train.py
    df_in['ph_neutral_diff'] = np.abs(df_in['ph'] - 7.0)
    df_in['Solids_Conductivity_ratio'] = df_in['Solids'] / (df_in['Conductivity'] + 1e-5)
    df_in['Chloramines_Sulfate_ratio'] = df_in['Chloramines'] / (df_in['Sulfate'] + 1e-5)
    df_in['Total_Chemical_Load'] = (
        df_in['Chloramines'] + df_in['Organic_carbon'] + (df_in['Trihalomethanes'] / 10.0)
    )
    scaled_features = scaler.transform(df_in)
    return raw, scaled_features

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    probability = None
    input_values = None

    if request.method == "POST":
        raw_data, scaled_X = prepare_input(request.form)
        pred = model.predict(scaled_X)[0]
        prob = model.predict_proba(scaled_X)[0][1]

        save_prediction(raw_data, pred, prob)
        prediction_result = int(pred)
        probability = round(float(prob) * 100, 2)
        input_values = raw_data

    return render_template(
        "index.html",
        metadata=metadata,
        prediction=prediction_result,
        probability=probability,
        input_values=input_values
    )

@app.route("/history")
def history():
    records = get_all_predictions()
    return render_template("history.html", records=records)

if __name__ == "__main__":
    app.run(debug=True)