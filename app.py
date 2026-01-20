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
# CUSTOM STYLING
# -----------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🌦️ Rainfall Prediction App")

col1, col2 = st.columns([2, 1])

with col1:
    temperature = st.number_input("Temperature (K)", 250.0, 320.0, 300.0)
    windspeed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 2.0)
    rain_prev1 = st.number_input("Yesterday's Rainfall (mm)", 0.0, 500.0, 0.0)
    month = st.number_input("Month (1-12)", 1, 12, 1)
    predict_btn = st.button("Predict 🌧️")

with col2:
    st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)

    if predict_btn:
        X_input = pd.DataFrame(
            [[temperature, windspeed, rain_prev1, month]],
            columns=['temperature', 'windspeed', 'rain_prev1', 'month']
        )

        rf_pred = rf_model.predict(X_input)[0]
        xgb_pred = xgb_model.predict(X_input)[0]

        max_pred = max(rf_pred, xgb_pred)

        if max_pred < 1.0:
            alert_class = "alert-low"
            alert_text = "☀️ Light Rainfall"
        elif max_pred < 3.0:
            alert_class = "alert-medium"
            alert_text = "🌦️ Moderate Rainfall"
        else:
            alert_class = "alert-high"
            alert_text = "🌧️ Heavy Rainfall — Bring an Umbrella!"

        st.markdown(f'<p class="pred-text">Random Forest Prediction (mm): {rf_pred:.3f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="pred-text">XGBoost Prediction (mm): {xgb_pred:.3f}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="{alert_class}">{alert_text}</p>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
# -----------------------------
# APP LAYOUT
# -----------------------------


