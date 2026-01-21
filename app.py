import streamlit as st
import numpy as np
import joblib

# Load models
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

st.set_page_config(page_title="Rainfall Prediction App", layout="wide")

# Default background (Light Rain)
bg_dynamic = "https://d2u0ktu8omkpf6.cloudfront.net/e0036137a0c69370e3e4909d4cd47cbe621cab64cbe866b9.jpg"

# CSS for background, no scroll, high contrast labels
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        overflow: hidden;
    }}

    .stApp {{
        background-image: url("{bg_dynamic}");
        background-size: cover;
        background-position: center;
    }}

    label {{
        font-size: 22px !important;
        font-weight: bold !important;
        color: white !important;
        background-color: rgba(0,0,0,0.6);
        padding: 6px 10px;
        border-radius: 8px;
    }}

    h1 {{
        color: white;
        text-align: center;
        background-color: rgba(0,0,0,0.6);
        padding: 10px;
        border-radius: 10px;
    }}

    .result-box {{
        color: white;
        font-size: 22px;
        font-weight: bold;
        background-color: rgba(0,0,0,0.6);
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1>🌧 Rainfall Prediction App</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("🌡 Temperature (Kelvin)", value=300.0)
    wind_speed = st.number_input("💨 Wind Speed (m/s)", value=5.0)
    yesterday_rain = st.number_input("🌧 Yesterday's Rainfall (mm)", value=20.0)
    month = st.number_input("📅 Month (1–12)", min_value=1, max_value=12, value=6)

with col2:
    st.markdown("<div class='result-box'>Prediction Results will appear here 👇</div>", unsafe_allow_html=True)

# Predict button
if st.button("Predict 🌦"):

    X = np.array([[temperature, wind_speed, yesterday_rain, month]])

    rf_pred = rf_model.predict(X)[0]
    xgb_pred = xgb_model.predict(X)[0]

    max_pred = max(rf_pred, xgb_pred)

    # Decide rainfall type + background
    if max_pred < 1.0:
        rainfall_type = "🌤 Light Rainfall"
        bg_dynamic = "https://d2u0ktu8omkpf6.cloudfront.net/e0036137a0c69370e3e4909d4cd47cbe621cab64cbe866b9.jpg"

    elif max_pred < 5.0:
        rainfall_type = "🌦 Moderate Rainfall"
        bg_dynamic = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4M0Kt3rSFoshV0ixydRW83zhRLTVTzi2suw&s"

    else:
        rainfall_type = "⛈ Heavy Rainfall"
        bg_dynamic = "https://pragativadi.com/wp-content/uploads/2025/06/IMD-Issues-Orange-Alert-Thunderstorm-Heavy-Rainfall-Likely-in-Odisha-Districts-Over-Next-Four-Days.jpg"

    # Update background after prediction
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_dynamic}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Show results
    st.markdown(
        f"""
        <div class="result-box">
        🌲 Random Forest Prediction: {rf_pred:.2f} mm <br>
        ⚡ XGBoost Prediction: {xgb_pred:.2f} mm <br><br>
        👉 Final Result: <b>{rainfall_type}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
