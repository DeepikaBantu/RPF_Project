import streamlit as st
import joblib
import gzip
import numpy as np
import requests

# -----------------------------
# DOWNLOAD MODELS
# -----------------------------
import gdown
import joblib
import gzip
import os

# RF model compressed
rf_gz_url = "https://drive.google.com/uc?id=1VU-BxyOFibyls7aP4R8rcWRL4WS9iiwh"
rf_gz_path = "rf_model_compressed.pkl.gz"
rf_pkl_path = "rf_model.pkl"

# Download compressed file if not exists
if not os.path.exists(rf_gz_path):
    gdown.download(rf_gz_url, rf_gz_path, quiet=False)

# Uncompress
if not os.path.exists(rf_pkl_path):
    with gzip.open(rf_gz_path, "rb") as f_in:
        with open(rf_pkl_path, "wb") as f_out:
            f_out.write(f_in.read())

# Load RF model
rf_model = joblib.load(rf_pkl_path)

# --- XGBoost model from GitHub ---
xgb_url = "https://raw.githubusercontent.com/DeepikaBantu/RPF_Project/main/xgb_model.pkl"
xgb_model = joblib.load("xgb_model.pkl")  # assuming already present locally or uploaded to repo

# --- Random Forest compressed model from Drive ---
rf_id = "1VU-BxyOFibyls7aP4R8rcWRL4WS9iiwh"  # compressed RF model (.pkl.gz)
session = requests.Session()
URL = "https://docs.google.com/uc?export=download"

response = session.get(URL, params={'id': rf_id}, stream=True)
confirm_token = None
for key, value in response.cookies.items():
    if key.startswith('download_warning'):
        confirm_token = value

if confirm_token:
    params = {'id': rf_id, 'confirm': confirm_token}
    response = session.get(URL, params=params, stream=True)

# Save compressed file
rf_gz_path = "rf_model_compressed.pkl.gz"
with open(rf_gz_path, "wb") as f:
    for chunk in response.iter_content(32768):
        if chunk:
            f.write(chunk)

# Decompress and load
with gzip.open(rf_gz_path, "rb") as f_in:
    rf_model = joblib.load(f_in)

# -----------------------------
# CUSTOM CSS FOR BRIGHT TEXT & LAYOUT
# -----------------------------
st.markdown("""
<style>
/* Full app container with background image */
.stApp > div:first-child {
    background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1350&q=80');
    background-size: cover;
    background-attachment: fixed;
    padding: 20px;
    border-radius: 10px;
}

/* Main content overlay */
.stApp {
    background: rgba(0,0,0,0.2);  /* very light overlay */
    padding: 15px;
    border-radius: 10px;
    color: #ffffff;
}

/* Headings and labels */
h1, h2, h3, h4, h5, h6, label {
    color: #ffffff !important;
}

/* Input text boxes */
input {
    color: #ffffff !important;
    background-color: rgba(0,0,0,0.3) !important;
}

/* Buttons styling */
.stButton>button {
    color: #ffffff;
    background-color: #4CAF50;  /* bright green button */
    font-size: 18px;
    font-weight: bold;
}

/* Prediction alert styling */
.alert-high { color: #ff4b4b; font-size: 24px; font-weight: bold; }
.alert-medium { color: #ffd700; font-size: 24px; font-weight: bold; }
.alert-low { color: #00ff00; font-size: 24px; font-weight: bold; }

/* Prediction panel on right */
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

# Input columns
col1, col2 = st.columns([2, 1])  # inputs on left, predictions on right

with col1:
    temperature = st.number_input("Temperature (K)", min_value=250.0, max_value=320.0, value=300.0)
    windspeed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=2.0)
    prev_rain = st.number_input("Yesterday's Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0)
    predict_btn = st.button("Predict 🌧️")

with col2:
    st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
    if predict_btn:
        # Prepare input
        X_input = np.array([[temperature, windspeed, prev_rain]])
        rf_pred = rf_model.predict(X_input)[0]
        xgb_pred = xgb_model.predict(X_input)[0]

        # Determine alert
        if max(rf_pred, xgb_pred) < 1.0:
            alert_class = "alert-low"
            alert_text = "☀️ Light Rainfall"
        elif max(rf_pred, xgb_pred) < 10.0:
            alert_class = "alert-medium"
            alert_text = "🌦️ Moderate Rainfall"
        else:
            alert_class = "alert-high"
            alert_text = "🌧️ Heavy Rainfall — Bring an Umbrella!"

        st.markdown(f"**Random Forest Prediction (mm):** {rf_pred:.3f}")
        st.markdown(f"**XGBoost Prediction (mm):** {xgb_pred:.3f}")
        st.markdown(f'<p class="{alert_class}">{alert_text}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

