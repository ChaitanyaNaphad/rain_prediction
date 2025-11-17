import streamlit as st
import numpy as np
import joblib
import requests
import os

# -------------------------------------------------------
# Firebase URL
# -------------------------------------------------------
FIREBASE_URL = "https://rain-prediction-e5448-default-rtdb.asia-southeast1.firebasedatabase.app"

# -------------------------------------------------------
# URLs of Model & Scaler stored in GitHub (RAW LINKS)
# -------------------------------------------------------
MODEL_URL = "https://raw.githubusercontent.com/ChaitanyaNaphad/rain_prediction/main/rain_model.pkl"
SCALER_URL = "https://raw.githubusercontent.com/ChaitanyaNaphad/rain_prediction/main/scaler.pkl"

# -------------------------------------------------------
# Download + Load Model & Scaler
# -------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():

    model_path = "rain_model.pkl"
    scaler_path = "scaler.pkl"

    # --- Download model file if not exists ---
    if not os.path.exists(model_path):
        r = requests.get(MODEL_URL)
        open(model_path, "wb").write(r.content)

    # --- Download scaler file if not exists ---
    if not os.path.exists(scaler_path):
        r = requests.get(SCALER_URL)
        open(scaler_path, "wb").write(r.content)

    # --- Load using joblib ---
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


# Load once (cached)
model, scaler = load_model_and_scaler()

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.title("🌧️ Rainfall Prediction Web App")
st.write("Predict **rainfall (mm)** using a trained model stored on GitHub & upload results to Firebase.")

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
# Predict Button
# -------------------------------------------------------
if st.button("Predict Rainfall"):

    X_new = np.array([[pressure, maxtemp, temparature, mintemp,
                       dewpoint, humidity, cloud, windspeed]])

    X_new_scaled = scaler.transform(X_new)
    prediction = float(model.predict(X_new_scaled)[0])

    st.success(f"🌧️ **Predicted Rainfall: {prediction:.2f} mm**")

    # -------------------------------------------------------
    # Upload to Firebase
    # -------------------------------------------------------
    data = {
        "pressure": float(pressure),
        "maxtemp": float(maxtemp),
        "temparature": float(temparature),
        "mintemp": float(mintemp),
        "dewpoint": float(dewpoint),
        "humidity": float(humidity),
        "cloud": float(cloud),
        "windspeed": float(windspeed),
        "prediction_mm": prediction,
        "status": "High" if prediction > 50 else "Normal"
    }

    firebase_endpoint = f"{FIREBASE_URL}/predictions.json"
    response = requests.post(firebase_endpoint, json=data)

    if response.status_code == 200:
        st.success("📡 Data uploaded to Firebase successfully!")
    else:
        st.error("❌ Failed to upload data to Firebase")
