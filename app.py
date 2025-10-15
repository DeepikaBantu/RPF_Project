# enhanced_app.py
import streamlit as st
import joblib
import pandas as pd

# Load trained models
import gdown
import joblib

# Download and load the XGBoost model
xgb_url = "https://drive.google.com/uc?id=1oRs0MGL4KDxjf8mX31dtAjKMvRbUkatS"
gdown.download(xgb_url, "xgb_model.pkl", quiet=False)
xgb_model = joblib.load("xgb_model.pkl")

# Download and load the Random Forest model
rf_url = "https://drive.google.com/uc?id=1AprnF_FHSmSHQL-tAvAZu5AMLD8MK-Ae"
gdown.download(rf_url, "rf_model.pkl", quiet=False)
rf_model = joblib.load("rf_model.pkl")

# Page setup
st.set_page_config(page_title="Rainfall Prediction System 🌧", layout="wide")

# Background and styling
import streamlit as st

st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
    }
    .stApp {
        background: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        color: #ffffff;  /* Make default text white */
    }
    h1, h2, h3, h4, h5, h6, label, .css-1aumxhk {
        color: #ffffff !important;  /* Force headings and labels to white */
    }
    .alert-high {
        color: #ff4b4b;
        font-size: 24px;
        font-weight: bold;
    }
    .alert-medium {
        color: #ffd700;
        font-size: 24px;
        font-weight: bold;
    }
    .alert-low {
        color: #00ff00;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🌧 Rainfall Prediction System")

# Layout: Left = Inputs, Right = Predictions
col1, col2 = st.columns([1,1])

with col1:
    st.header("Enter Current Weather Conditions")
    temperature = st.number_input("Temperature (K)", min_value=200.0, max_value=350.0, value=288.0, step=0.1)
    windspeed = st.number_input("Windspeed (m/s)", min_value=0.0, max_value=50.0, value=3.0, step=0.1)
    rain_prev1 = st.number_input("Previous Day Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.2, step=0.01)
    month = st.slider("Month", min_value=1, max_value=12, value=1)

    # DataFrame for prediction
    input_df = pd.DataFrame({
        'temperature': [temperature],
        'windspeed': [windspeed],
        'rain_prev1': [rain_prev1],
        'month': [month]
    })

with col2:
    st.header("Predicted Rainfall")
    if st.button("Predict Rainfall"):
        rf_pred = rf_model.predict(input_df)[0]
        xgb_pred = xgb_model.predict(input_df)[0]

        # Display predictions
        st.metric("Random Forest Prediction (mm)", f"{rf_pred:.3f}")
        st.metric("XGBoost Prediction (mm)", f"{xgb_pred:.3f}")

        # Heavy rainfall alert
        avg_pred = (rf_pred + xgb_pred)/2

        if avg_pred >= 20:
            st.markdown(f"<p class='alert-high'>⚠️ Heavy Rainfall Alert!</p>", unsafe_allow_html=True)
        elif avg_pred >= 10:
            st.markdown(f"<p class='alert-medium'>🌧 Moderate Rainfall Expected</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='alert-low'>☀️ Light Rainfall</p>", unsafe_allow_html=True)

st.markdown("<br><br><p style='color:white;'>Developed by Deepika Bantu | AI & Machine Learning</p>", unsafe_allow_html=True)



