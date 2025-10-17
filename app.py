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
st.markdown("""
<style>
.stApp > div:first-child { 
    background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1350&q=80'); 
    background-size: cover; background-attachment: fixed; padding: 20px; border-radius: 10px; 
}
.stApp { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px; color: #ffffff; }
h1, h2, h3, h4, h5, h6, label { color: #ffffff !important; }
input { color: #ffffff !important; background-color: rgba(0,0,0,0.3) !important; }
.stButton>button { color: #ffffff; background-color: #4CAF50; font-size: 18px; font-weight: bold; }
.alert-high { color: #ff4b4b; font-size: 36px; font-weight: bold; }
.alert-medium { color: #ffd700; font-size: 36px; font-weight: bold; }
.alert-low { color: #00ff00; font-size: 36px; font-weight: bold; }
.prediction-panel { background-color: rgba(0,0,0,0.35); padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# APP LAYOUT
# -----------------------------
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
        # Create DataFrame with correct feature names
        X_input = pd.DataFrame([[temperature, windspeed, rain_prev1, month]],
                               columns=['temperature', 'windspeed', 'rain_prev1', 'month'])

        # Predictions
        rf_pred = rf_model.predict(X_input)[0]
        xgb_pred = xgb_model.predict(X_input)[0]

        # Determine alert
        max_pred = max(rf_pred, xgb_pred)
        if max_pred < 1.0:
            alert_class = "alert-low"
            alert_text = "☀️ Light Rainfall"
        elif max_pred < 10.0:
            alert_class = "alert-medium"
            alert_text = "🌦️ Moderate Rainfall"
        else:
            alert_class = "alert-high"
            alert_text = "🌧️ Heavy Rainfall — Bring an Umbrella!"

        st.markdown(f"**Random Forest Prediction (mm):** {rf_pred:.3f}")
        st.markdown(f"**XGBoost Prediction (mm):** {xgb_pred:.3f}")
        st.markdown(f'<p class="{alert_class}">{alert_text}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
