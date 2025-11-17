import streamlit as st
import pandas as pd
import joblib
import requests
import datetime

# ------------------------------
# LOAD MODEL & SCALER
# ------------------------------
model_path = r"E:\python\ml_project\rain_model.pkl"
scaler_path = r"E:\python\ml_project\scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ------------------------------
# FIREBASE CONFIG
# ------------------------------
FIREBASE_URL = "https://rain-prediction-e5448-default-rtdb.asia-southeast1.firebasedatabase.app/"  
# IMPORTANT — must end with '/'

# ------------------------------
# APP UI
# ------------------------------
st.title("🌧️ Rainfall Prediction System")
st.write("Enter the values below to predict rainfall")

rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 10.0)
maxtemp = st.number_input("Max Temperature (°C)", 0.0, 60.0, 30.0)
mintemp = st.number_input("Min Temperature (°C)", 0.0, 60.0, 20.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 200.0, 10.0)

if st.button("Predict"):
    try:
        # ------------------------------
        # PREPARE INPUT
        # ------------------------------
        input_data = pd.DataFrame([[rainfall, maxtemp, mintemp, humidity, wind_speed]],
                                  columns=['rainfall', 'maxtemp', 'mintemp', 'humidity', 'wind_speed'])

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        st.success(f"🌧️ Predicted Rainfall: **{prediction:.2f} mm**")

        # ------------------------------
        # PREPARE FIREBASE DATA
        # ------------------------------
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "rainfall": rainfall,
            "maxtemp": maxtemp,
            "mintemp": mintemp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "predicted_rainfall": float(prediction)
        }

        # ------------------------------
        # POST TO FIREBASE
        # ------------------------------
        firebase_endpoint = FIREBASE_URL + "predictions.json"
        response = requests.post(firebase_endpoint, json=data)

        st.write("📡 Firebase Upload Status:", response.status_code)
        st.write("🔍 Firebase Response:", response.text)

        if response.status_code == 200:
            st.success("✅ Prediction uploaded to Firebase!")
        else:
            st.error("❌ Failed to upload to Firebase. Check logs above.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
