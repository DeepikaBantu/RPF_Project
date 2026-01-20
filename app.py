import streamlit as st
import joblib
import pandas as pd
import gdown
import os
import numpy as np

# -----------------------------
# DOWNLOAD AND LOAD MODELS
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
# APP TITLE
# -----------------------------
st.title("🌦️ Rainfall Prediction App")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🌡️ Temperature (Kelvin)")
    temperature = st.number_input("", 250.0, 320.0, 300.0)

    st.markdown("### 💨 Wind Speed (m/s)")
    windspeed = st.number_input("", 0.0, 20.0, 2.0)

    st.markdown("### 🌧️ Yesterday's Rainfall (mm)")
    rain_prev1 = st.number_input("", 0.0, 500.0, 0.0)

    st.markdown("### 📅 Month (1–12)")
    month = st.number_input("", 1, 12, 1)

    predict_btn = st.button("🔮 Predict Rainfall")

with col2:
    result_text = ""
    bg_dynamic = ""

    if predict_btn:
        X_input = pd.DataFrame(
            [[temperature, windspeed, rain_prev1, month]],
            columns=['temperature', 'windspeed', 'rain_prev1', 'month']
        )

        rf_pred = rf_model.predict(X_input)[0]
        xgb_pred = xgb_model.predict(X_input)[0]
        max_pred = max(rf_pred, xgb_pred)

        # 🔹 Dynamic background based on rainfall
        if max_pred < 1.0:
            result_text = "☀️ Light Rainfall"
            bg_dynamic = "https://d2u0ktu8omkpf6.cloudfront.net/e0036137a0c69370e3e4909d4cd47cbe621cab64cbe866b9.jpg"

        elif max_pred < 3.0:
            result_text = "🌦️ Moderate Rainfall"
            bg_dynamic = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4M0Kt3rSFoshV0ixydRW83zhRLTVTzi2suw&s"

        else:
            result_text = "🌧️ Heavy Rainfall — Bring an Umbrella!"
            bg_dynamic = "https://pragativadi.com/wp-content/uploads/2025/06/IMD-Issues-Orange-Alert-Thunderstorm-Heavy-Rainfall-Likely-in-Odisha-Districts-Over-Next-Four-Days.jpg"

        st.markdown("### 📊 Prediction Results")
        st.markdown(f"**Random Forest Prediction (mm):** {rf_pred:.3f}")
        st.markdown(f"**XGBoost Prediction (mm):** {xgb_pred:.3f}")
        st.markdown(f"**{result_text}**")

# -----------------------------
# CUSTOM STYLING WITH BACKGROUND
# -----------------------------
st.markdown(f"""
<style>
/* Dynamic Background */
.stApp > div:first-child {{
    background-image: url('{bg_dynamic}');
    background-size: cover;
    background-attachment: fixed;
    padding: 20px;
    border-radius: 10px;
}}
/* Dark overlay */
.stApp {{
    background: rgba(0,0,0,0.35);
    padding: 20px;
    border-radius: 10px;
}}

/* Big Title */
h1 {{
    color: white !important;
    font-size: 48px !important;
    text-align: center;
}}

/* Big Labels */
label {{
    color: white !important;
    font-size: 24px !important;
    font-weight: bold;
}}

/* Button */
.stButton>button {{
    font-size: 22px;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
}}

</style>
""", unsafe_allow_html=True)
