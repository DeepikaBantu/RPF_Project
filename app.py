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
# DEFAULT BACKGROUND
# -----------------------------
bg_image = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1350&q=80"

# -----------------------------
# APP LAYOUT
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

# -----------------------------
# PREDICTION PANEL
# -----------------------------
with col2:

    result_text = ""
    alert_class = ""
    bg_dynamic = bg_image

    if predict_btn:

        X_input = pd.DataFrame(
            [[temperature, windspeed, rain_prev1, month]],
            columns=['temperature', 'windspeed', 'rain_prev1', 'month']
        )

        rf_pred = rf_model.predict(X_input)[0]
        xgb_pred = xgb_model.predict(X_input)[0]

        max_pred = max(rf_pred, xgb_pred)

        # Light Rainfall
        if max_pred < 1.0:
            alert_class = "alert-low"
            result_text = "☀️ Light Rainfall"
            bg_dynamic = "https://images.unsplash.com/photo-1502082553048-f009c37129b9?auto=format&fit=crop&w=1350&q=80"

        # Moderate Rainfall
        elif max_pred < 3.0:
            alert_class = "alert-medium"
            result_text = "🌦️ Moderate Rainfall"
            bg_dynamic = "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?auto=format&fit=crop&w=1350&q=80"

        # Heavy Rainfall
        else:
            alert_class = "alert-high"
            result_text = "🌧️ Heavy Rainfall — Bring an Umbrella!"
            bg_dynamic = "https://images.unsplash.com/photo-1509223197845-458d87318791?auto=format&fit=crop&w=1350&q=80"

        st.markdown("### 📊 Prediction Results")
        st.markdown(f"**Random Forest Prediction (mm):** {rf_pred:.3f}")
        st.markdown(f"**XGBoost Prediction (mm):** {xgb_pred:.3f}")
        st.markdown(f'<p class="{alert_class}">{result_text}</p>', unsafe_allow_html=True)

# -----------------------------
# CUSTOM STYLING WITH DYNAMIC BACKGROUND
# -----------------------------
st.markdown(f"""
<style>
.stApp > div:first-child {{
    background-image: url('{bg_dynamic}');
    background-size: cover;
    background-attachment: fixed;
    padding: 20px;
    border-radius: 10px;
}}

.stApp {{
    background: rgba(0,0,0,0.3);
    padding: 20px;
    border-radius: 10px;
}}

h1 {{
    color: white;
    font-size: 48px;
    text-align: center;
}}

label {{
    font-size: 22px !important;
    font-weight: bold;
    color: white !important;
}}

.alert-high {{
    color: #ff4b4b;
    font-size: 36px;
    font-weight: bold;
}}

.alert-medium {{
    color: #ffd700;
    font-size: 36px;
    font-weight: bold;
}}

.alert-low {{
    color: #00ff00;
    font-size: 36px;
    font-weight: bold;
}}

.stButton>button {{
    font-size: 20px;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
}}

</style>
""", unsafe_allow_html=True)
