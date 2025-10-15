import streamlit as st
import joblib
import numpy as np
import gdown
import os

# -----------------------------
# GOOGLE DRIVE LINKS
# -----------------------------
xgb_url = "https://drive.google.com/uc?id=/1oRs0MGL4KDxjf8mX31dtAjKMvRbUkatS"
rf_url = "https://drive.google.com/uc?id=1AprnF_FHSmSHQL-tAvAZu5AMLD8MK-Ae"

# -----------------------------
# DOWNLOAD MODELS IF NOT EXIST
# -----------------------------
if not os.path.exists("xgb_model.pkl"):
    gdown.download(xgb_url, "xgb_model.pkl", quiet=False)

if not os.path.exists("rf_model.pkl"):
    gdown.download(rf_url, "rf_model.pkl", quiet=False)

# LOAD MODELS
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# -----------------------------
# CSS FOR BRIGHT TEXT & BACKGROUND
# -----------------------------
st.markdown("""
<style>
.stApp > div:first-child {
    background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1350&q=80');
    background-size: cover;
    background-attachment: fixed;
    padding: 20px;
    border-radius: 10px;
}

.stApp { background: transparent; color: #ffffff; padding: 0px; }

h1, h2, h3, h4, h5, h6, label { color: #ffffff !important; }

input { color: #ffffff !important; background-color: rgba(0,0,0,0.25) !important; }

.stButton>button { color: #ffffff; background-color: #4CAF50; font-size: 18px; font-weight: bold; }

.prediction-panel { background-color: rgba(0,0,0,0.4); padding: 20px; border-radius: 10px; }

.alert-high { color: #ff4b4b; font-size: 24px; font-weight: bold; }
.alert-medium { color: #ffd700; font-size: 24px; font-weight: bold; }
.alert-low { color: #00ff00; font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# APP LAYOUT
# -----------------------------
st.title("Rainfall Prediction 🌦️")
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input Weather Data")
    temperature = st.number_input("Temperature (K)", min_value=250.0, max_value=320.0, value=300.0)
    windspeed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=2.0)
    prev_rain = st.number_input("Yesterday's Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0)
    predict_btn = st.button("Predict 🌧️")

with col2:
    st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
    if predict_btn:
        X_input = np.array([[temperature, windspeed, prev_rain]])
        rf_pred = rf_model.predict(X_input)[0]
        xgb_pred = xgb_model.predict(X_input)[0]

        max_pred = max(rf_pred, xgb_pred)
        if max_pred < 1.0:
            alert_class = "alert-low"
            alert_text = "☀️ Light Rainfall – No special precautions needed."
        elif max_pred < 10.0:
            alert_class = "alert-medium"
            alert_text = "🌦️ Moderate Rainfall – You might need a light raincoat."
        else:
            alert_class = "alert-high"
            alert_text = "🌧️ Heavy Rainfall – Bring an umbrella and stay safe!"

        st.markdown(f"**Random Forest Prediction (mm):** {rf_pred:.3f}")
        st.markdown(f"**XGBoost Prediction (mm):** {xgb_pred:.3f}")
        st.markdown(f'<p class="{alert_class}">{alert_text}</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
