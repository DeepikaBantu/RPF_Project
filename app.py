import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import os

st.set_page_config(page_title="Rainfall Prediction App", layout="wide")

# -----------------------------
# DOWNLOAD MODELS
# -----------------------------
rf_model_url = "https://drive.google.com/uc?id=1AprnF_FHSmSHQL-tAvAZu5AMLD8MK-Ae"
xgb_model_url = "https://drive.google.com/uc?id=1oRs0MGL4KDxjf8mX31dtAjKMvRbUkatS"

rf_model_path = "rf_model.pkl"
xgb_model_path = "xgb_model.pkl"

if not os.path.exists(rf_model_path):
    st.info("🔽 Downloading Random Forest model...")
    gdown.download(rf_model_url, rf_model_path, quiet=False)

if not os.path.exists(xgb_model_path):
    st.info("🔽 Downloading XGBoost model...")
    gdown.download(xgb_model_url, xgb_model_path, quiet=False)

rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)

# -----------------------------
# DEFAULT BACKGROUND
# -----------------------------
bg_dynamic = "https://d2u0ktu8omkpf6.cloudfront.net/e0036137a0c69370e3e4909d4cd47cbe621cab64cbe866b9.jpg"

# -----------------------------
# CUSTOM STYLE
# -----------------------------
st.markdown(f"""
<style>
html, body, [class*="css"] {{
    overflow: hidden;
}}

.stApp {{
    background-image: url("{bg_dynamic}");
    background-size: cover;
    background-position: center;
}}

h1 {{
    color: white;
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    background-color: rgba(0,0,0,0.75);
    padding: 15px;
    border-radius: 12px;
}}

label {{
    font-size: 22px !important;
    font-weight: bold !important;
    color: white !important;
    background-color: rgba(0,0,0,0.7);
    padding: 6px 10px;
    border-radius: 8px;
}}

.result-box {{
    color: white;
    font-size: 22px;
    font-weight: bold;
    background-color: rgba(0,0,0,0.75);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}}

.stButton>button {{
    font-size: 20px;
    font-weight: bold;
    padding: 10px 30px;
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1>🌧 RAINFALL PREDICTION APP</h1>", unsafe_allow_html=True)

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    temperature = st.number_input("🌡 Temperature (Kelvin)", value=300.0)
    wind_speed = st.number_input("💨 Wind Speed (m/s)", value=5.0)
    yesterday_rain = st.number_input("🌧 Yesterday's Rainfall (mm)", value=20.0)
    month = st.number_input("📅 Month (1–12)", min_value=1, max_value=12, value=6)

    predict_btn = st.button("Predict 🌦")

with col2:
    result_placeholder = st.empty()

# -----------------------------
# PREDICTION LOGIC
# -----------------------------
if predict_btn:

    X_input = pd.DataFrame(
        [[temperature, wind_speed, yesterday_rain, month]],
        columns=['temperature', 'windspeed', 'rain_prev1', 'month']
    )

    rf_pred = rf_model.predict(X_input)[0]
    xgb_pred = xgb_model.predict(X_input)[0]

    max_pred = max(rf_pred, xgb_pred)

    # Decision + Message + Background
    if max_pred < 1.0:
        rainfall_type = "🌤 Light Rainfall"
        message = "😊 Weather is safe. Light rain expected."
        bg_dynamic = "https://d2u0ktu8omkpf6.cloudfront.net/e0036137a0c69370e3e4909d4cd47cbe621cab64cbe866b9.jpg"

    elif max_pred < 5.0:
        rainfall_type = "🌦 Moderate Rainfall"
        message = "☔ Bring an umbrella. Drive safely!"
        bg_dynamic = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4M0Kt3rSFoshV0ixydRW83zhRLTVTzi2suw&s"

    else:
        rainfall_type = "⛈ Heavy Rainfall"
        message = "🚨 Heavy rain! Avoid going outside and stay safe."
        bg_dynamic = "https://pragativadi.com/wp-content/uploads/2025/06/IMD-Issues-Orange-Alert-Thunderstorm-Heavy-Rainfall-Likely-in-Odisha-Districts-Over-Next-Four-Days.jpg"

    # Update background dynamically
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{bg_dynamic}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Show results in RIGHT PANEL
    with col2:
        result_placeholder.markdown(f"""
        <div class="result-box">
        🌲 Random Forest Prediction: {rf_pred:.2f} mm <br>
        ⚡ XGBoost Prediction: {xgb_pred:.2f} mm <br><br>
        👉 Final Result: <b>{rainfall_type}</b> <br><br>
        💡 <b>Alert:</b> {message}
        </div>
        """, unsafe_allow_html=True)
