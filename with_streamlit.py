import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# -------------------------------------------------------
# 🔥 Firebase Base URL
# -------------------------------------------------------
FIREBASE_URL = "https://rain-prediction-e5448-default-rtdb.asia-southeast1.firebasedatabase.app"

# -------------------------------------------------------
# 1️⃣ Load Pretrained Model + Scaler
# -------------------------------------------------------

MODEL_URL = "https://raw.githubusercontent.com/ChaitanyaNaphad/rain_prediction/main/rain_model.pkl"
SCALER_URL = "https://raw.githubusercontent.com/ChaitanyaNaphad/rain_prediction/main/scaler.pkl"

# Download and load PKL files directly from GitHub Raw
gbr = joblib.load(MODEL_URL)
scaler = joblib.load(SCALER_URL)

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.title("🌧️ Rainfall Prediction Web App")
st.write("Predict **rainfall (mm)** using a pretrained ML model and save results to Firebase.")

st.subheader("🌡️ Enter Weather Parameters")

pressure = st.number_input("Pressure", step=0.1)
maxtemp = st.number_input("Max Temperature", step=0.1)
temparature = st.number_input("Temperature", step=0.1)
mintemp = st.number_input("Min Temperature", step=0.1)
dewpoint = st.number_input("Dew Point", step=0.1)
humidity = st.number_input("Humidity (%)", step=0.1)
cloud = st.number_input("Cloud (%)", step=0.1)
windspeed = st.number_input("Wind Speed", step=0.1)

# -------------------------------------------------------
# 2️⃣ Predict Rainfall + Upload to Firebase
# -------------------------------------------------------
if st.button("Predict Rainfall"):

    X_new = np.array([
        [pressure, maxtemp, temparature, mintemp,
         dewpoint, humidity, cloud, windspeed]
    ])

    X_new_scaled = scaler.transform(X_new)
    y_new = gbr.predict(X_new_scaled)

    predicted_value = float(y_new[0])

    st.success(f"🌧️ **Predicted Rainfall: {predicted_value:.2f} mm**")

    data = {
        "pressure": float(pressure),
        "maxtemp": float(maxtemp),
        "temparature": float(temparature),
        "mintemp": float(mintemp),
        "dewpoint": float(dewpoint),
        "humidity": float(humidity),
        "cloud": float(cloud),
        "windspeed": float(windspeed),
        "prediction_mm": predicted_value,
        "status": "High" if predicted_value > 50 else "Normal"
    }

    firebase_endpoint = f"{FIREBASE_URL}/predictions.json"
    response = requests.post(firebase_endpoint, json=data)

    if response.status_code == 200:
        st.success("📡 Prediction successfully uploaded to Firebase!")
    else:
        st.error("❌ Failed to upload prediction to Firebase.")
