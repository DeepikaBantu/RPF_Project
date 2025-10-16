import streamlit as st
import gdown
import joblib
import numpy as np
import requests
import os

# -----------------------------
# LOAD XGBOOST MODEL FROM GITHUB RAW LINK
# -----------------------------
xgb_github_url = "https://raw.githubusercontent.com/DeepikaBantu/RPF_Project/refs/heads/main/xgb_model.pkl"
xgb_model_path = "xgb_model.pkl"
res = requests.get(xgb_github_url)
if res.status_code == 200:
    with open(xgb_model_path, "wb") as f:
        f.write(res.content)
    xgb_model = joblib.load(xgb_model_path)
else:
    st.error("❌ Could not load XGBoost model from GitHub. Check raw URL.")
    st.stop()

# -----------------------------
# LOAD RANDOM FOREST MODEL FROM DRIVE
# -----------------------------
rf_url = "https://drive.google.com/uc?id=1AprnF_FHSmSHQL-tAvAZu5AMLD8MK-Ae"
rf_model_path = "rf_model.pkl"
# Only download if not already present
if not os.path.exists(rf_model_path):
    gdown.download(rf_url, rf_model_path, quiet=False)
rf_model = joblib.load(rf_model_path)

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

.stApp {
    background: rgba(0,0,0,0.2);
    padding: 15px;
    border-radius: 10px;
    color: #ffffff;
}

h1, h2, h3, h4, h5, h6, label {
    color: #ffffff !important;
}

input {
    color: #ffffff !important;
    background-color: rgba(0,0,0,0.3) !important;
}

.stButton>button {
    color: #ffffff;
    background-color: #4CAF50;
    font-size: 18px;
    font-weight: bold;
}

.alert-high { color: #ff4b4b; font-size: 24px; font-weight: bold; }
.alert-medium { color: #ffd700; font-size: 24px; font-weight: bold; }
.alert-low { color: #00ff00; font-size: 24px; font-weight: bold; }

.prediction-panel {
    background-color: rgba(0,0,0,0.35);
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# APP LAYOUT
# -----------------------------
st.title("Rainfall Prediction 🌦️")

col1, col2 = st.columns([2, 1])

with col1:
    temperature = st.number_input("Temperature (K)", min_value=250.0, max_value=320.0, value=300.0)
    windspeed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=2.0)
    prev_rain = st.number_input("Yesterday's Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0)
    predict_btn = st.button("Predict 🌧️")

with col2:
    st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
    if predict_btn:
        X_input = np.array([[temperature, windspeed, prev_rain]])
        # Predictions
        try:
            rf_pred = rf_model.predict(X_input)[0]
            xgb_pred = xgb_model.predict(X_input)[0]
        except Exception as e:
            st.error("❌ Prediction error: " + str(e))
            st.stop()

        maximum = max(rf_pred, xgb_pred)
        if maximum < 1.0:
            alert_class = "alert-low"
            alert_text = "☀️ Light Rainfall – No need umbrella."
        elif maximum < 10.0:
            alert_class = "alert-medium"
            alert_text = "🌦️ Moderate Rainfall – You might need an umbrella."
        else:
            alert_class = "alert-high"
            alert_text = "🌧️ Heavy Rainfall – Bring umbrella and stay safe!"

        st.markdown(f"**Random Forest (mm):** {rf_pred:.3f}")
        st.markdown(f"**XGBoost (mm):** {xgb_pred:.3f}")
        st.markdown(f'<p class="{alert_class}">{alert_text}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
