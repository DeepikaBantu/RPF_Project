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

@st.cache_resource
def load_models():
    rf = joblib.load(rf_model_path)
    xgb = joblib.load(xgb_model_path)
    return rf, xgb

rf_model, xgb_model = load_models()

# -----------------------------
# DEFAULT BACKGROUND
# -----------------------------
bg_dynamic = "https://d2u0ktu8omkpf6.cloudfront.net/e0036137a0c69370e3e4909d4cd47cbe621cab64cbe866b9.jpg"

# -----------------------------
# RAIN ANIMATION FUNCTION
# -----------------------------
def rain_animation(level):
    if level == "light":
        drops = 50
    elif level == "moderate":
        drops = 120
    else:
        drops = 250

    html = "<div class='rain'>"
    for i in range(drops):
        html += "<div class='drop'></div>"
    html += "</div>"

    return f"""
    <style>
    .rain {{
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        pointer-events: none;
        z-index: 1;
    }}
    .drop {{
        position: absolute;
        bottom: 100%;
        width: 2px;
        height: 15px;
        background: rgba(255,255,255,0.8);
        animation: fall linear infinite;
    }}
    @keyframes fall {{
        to {{
            transform: translateY(110vh);
        }}
    }}
    .drop {{
        left: {np.random.randint(0,100)}%;
        animation-duration: {np.random.uniform(0.5,1.5)}s;
        animation-delay: {np.random.uniform(0,2)}s;
    }}
    </style>
    {html}
    """

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
    color: #0B2C6B;
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    background-color: rgba(0,0,0,0.75);
    padding: 15px;
    border-radius: 12px;
    text-shadow: 2px 2px 6px black;
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

    # -----------------------------
    # DECISION + SOUND + ANIMATION
    # -----------------------------
    if max_pred < 2.0:
        rainfall_type = "🌤 Light Rainfall"
        message = "😊 Weather is safe. Light rain expected."
        rain_effect = "light"
        rain_sound = "https://www.soundjay.com/nature/rain-01.mp3"

    elif max_pred < 3.5:
        rainfall_type = "🌦 Moderate Rainfall"
        message = "☔ Bring an umbrella. Drive safely!"
        rain_effect = "moderate"
        rain_sound = "https://www.soundjay.com/nature/rain-03.mp3"

    else:
        rainfall_type = "⛈ Heavy Rainfall"
        message = "🚨 Heavy rain! Avoid going outside and stay safe."
        rain_effect = "heavy"
        rain_sound = "https://www.soundjay.com/nature/thunder-01.mp3"

    # -----------------------------
    # SHOW RAIN ANIMATION
    # -----------------------------
    st.markdown(rain_animation(rain_effect), unsafe_allow_html=True)

    # -----------------------------
    # PLAY RAIN SOUND
    # -----------------------------
    st.audio(rain_sound, autoplay=True)

    # -----------------------------
    # SHOW RESULTS
    # -----------------------------
    with col2:
        result_placeholder.markdown(f"""
        <div class="result-box">
        🌲 Random Forest Prediction: {rf_pred:.2f} mm <br>
        ⚡ XGBoost Prediction: {xgb_pred:.2f} mm <br><br>
        👉 Final Result: <b>{rainfall_type}</b> <br><br>
        💡 <b>Alert:</b> {message}
        </div>
        """, unsafe_allow_html=True)
