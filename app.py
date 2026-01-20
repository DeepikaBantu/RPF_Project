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
# -----------------------------
# APP TITLE
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🌦️ Rainfall Prediction App</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)

    st.markdown("🌡️ <span class='lbl'>Temperature (Kelvin)</span>", unsafe_allow_html=True)
    temperature = st.number_input("", 250.0, 320.0, 300.0, key="t")

    st.markdown("💨 <span class='lbl'>Wind Speed (m/s)</span>", unsafe_allow_html=True)
    windspeed = st.number_input("", 0.0, 20.0, 5.0, key="w")

    st.markdown("🌧️ <span class='lbl'>Yesterday's Rainfall (mm)</span>", unsafe_allow_html=True)
    rain_prev1 = st.number_input("", 0.0, 500.0, 100.0, key="r")

    st.markdown("📅 <span class='lbl'>Month (1–12)</span>", unsafe_allow_html=True)
    month = st.number_input("", 1, 12, 7, key="m")

    predict_btn = st.button("🔮 Predict Rainfall")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)

    if predict_btn:
        X_input = pd.DataFrame(
            [[temperature, windspeed, rain_prev1, month]],
            columns=['temperature', 'windspeed', 'rain_prev1', 'month']
        )

        rf_pred = rf_model.predict(X_input)[0]
        xgb_pred = xgb_model.predict(X_input)[0]
        max_pred = max(rf_pred, xgb_pred)

        # Background selection
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
        st.markdown(f"<h3 style='color:#00ffcc;'>{result_text}</h3>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# CUSTOM STYLING (NO SCROLL + HIGH CONTRAST)
# -----------------------------
st.markdown(f"""
<style>

/* Remove scroll bar */
body {{
    overflow: hidden;
}}

/* Dynamic Background */
.stApp > div:first-child {{
    background-image: url('{bg_dynamic}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    height: 100vh;
    padding: 10px;
}}

/* Dark overlay panel */
.input-panel {{
    background: rgba(0,0,0,0.55);
    padding: 18px;
    border-radius: 15px;
}}

.result-panel {{
    background: rgba(0,0,0,0.55);
    padding: 18px;
    border-radius: 15px;
    color: white;
}}

/* Labels */
.lbl {{
    color: #ffffff;
    font-size: 22px;
    font-weight: bold;
    text-shadow: 2px 2px 5px black;
}}

/* Title */
h1 {{
    color: white;
    font-size: 46px;
    text-shadow: 3px 3px 6px black;
}}

/* Button */
.stButton>button {{
    width: 100%;
    font-size: 22px;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}}

</style>
""", unsafe_allow_html=True)

