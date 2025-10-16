import streamlit as st
import joblib
import numpy as np
import requests

# --- XGBoost model from GitHub ---
xgb_url = "https://raw.githubusercontent.com/DeepikaBantu/RPF_Project/main/xgb_model.pkl"
xgb_model_path = "xgb_model.pkl"
xgb_model = joblib.load(xgb_model_path)

# --- Random Forest model from Google Drive (direct download link) ---
rf_url = "https://drive.google.com/uc?export=download&id=1AprnF_FHSmSHQL-tAvAZu5AMLD8MK-Ae"
rf_model_path = "rf_model.pkl"

# Download from Drive manually using requests (Streamlit Cloud safe)
try:
    r = requests.get(rf_url, allow_redirects=True)
    with open(rf_model_path, "wb") as f:
        f.write(r.content)
    rf_model = joblib.load(rf_model_path)
except Exception as e:
    st.error(f"Could not load RF model from Drive: {e}")
    st.stop()

# --- UI and styling ---
st.markdown(
    """
    <style>
        .alert-high {color:#ff4b4b;font-size:24px;font-weight:bold;}
        .alert-medium {color:#ffd700;font-size:24px;font-weight:bold;}
        .alert-low {color:#00ff00;font-size:24px;font-weight:bold;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌦️ Rainfall Prediction App")

temperature = st.number_input("Temperature (K)", 250.0, 320.0, 300.0)
windspeed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 2.0)
prev_rain = st.number_input("Yesterday's Rainfall (mm)", 0.0, 500.0, 0.0)

if st.button("Predict 🌧️"):
    X_input = np.array([[temperature, windspeed, prev_rain]])
    rf_pred = rf_model.predict(X_input)[0]
    xgb_pred = xgb_model.predict(X_input)[0]
    rain_pred = max(rf_pred, xgb_pred)

    if rain_pred < 1.0:
        st.markdown(f'<p class="alert-low">☀️ Light Rainfall ({rain_pred:.2f} mm)</p>', unsafe_allow_html=True)
    elif rain_pred < 10.0:
        st.markdown(f'<p class="alert-medium">🌦️ Moderate Rainfall ({rain_pred:.2f} mm)</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="alert-high">🌧️ Heavy Rainfall ({rain_pred:.2f} mm) — Bring an Umbrella! ☂️</p>', unsafe_allow_html=True)
