import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
# 🔥 Firebase Base URL
# -------------------------------------------------------
FIREBASE_URL = "https://rain-prediction-e5448-default-rtdb.asia-southeast1.firebasedatabase.app"

# -------------------------------------------------------
# 1️⃣ Load and Prepare Data
# -------------------------------------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/ChaitanyaNaphad/rain_prediction/main/Final_Rainfall_mm.csv"
)

cols = ['pressure','maxtemp','temparature','mintemp','dewpoint',
        'humidity','cloud','windspeed','rainfall']

for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

X = df[['pressure','maxtemp','temparature','mintemp','dewpoint',
        'humidity','cloud','windspeed']]
y = df['rainfall']

imputer_X = SimpleImputer(strategy='mean')
imputer_y = SimpleImputer(strategy='mean')
X = imputer_X.fit_transform(X)
y = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
gbr.fit(X_train_scaled, y_train)

y_pred = gbr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.title("🌧️ Rainfall Prediction Web App")
st.write("Predict **rainfall (mm)** using ML and save results to Firebase.")

st.subheader("📊 Model Performance")
st.write(f"**Mean Squared Error:** {float(mse):.4f}")
st.write(f"**R² Score:** {float(r2):.4f}")

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

    X_new = np.array([[pressure, maxtemp, temparature, mintemp,
                       dewpoint, humidity, cloud, windspeed]])

    X_new_scaled = scaler.transform(X_new)
    y_new = gbr.predict(X_new_scaled)

    predicted_value = float(y_new[0])   # convert to normal number

    st.success(f"🌧️ **Predicted Rainfall: {predicted_value:.2f} mm**")

    # ------------------------------------------
    # Prepare clean numeric Firebase data
    # ------------------------------------------
    data = {
        "pressure": float(pressure),
        "maxtemp": float(maxtemp),
        "temparature": float(temparature),
        "mintemp": float(mintemp),
        "dewpoint": float(dewpoint),
        "humidity": float(humidity),
        "cloud": float(cloud),
        "windspeed": float(windspeed),
        "prediction_mm": predicted_value,     # clean number
        "status": "High" if predicted_value > 50 else "Normal"
    }

    # ------------------------------------------
    # Upload to Firebase
    # ------------------------------------------
    firebase_endpoint = f"{FIREBASE_URL}/predictions.json"
    response = requests.post(firebase_endpoint, json=data)

    if response.status_code == 200:
        st.success("📡 Prediction successfully uploaded to Firebase!")
    else:
        st.error("❌ Failed to upload prediction to Firebase.")
