import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "accident_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "encoders.pkl"

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SafeRoad AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Result */
.result-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    background: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.10);
}

/* Small text */
.small-text {
    color: #777;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🚦 SafeRoad AI")

    st.write(
        "An AI-powered road accident severity prediction system."
    )

    st.divider()

    st.subheader("📌 How it works")

    st.write("1️⃣ Enter road conditions")
    st.write("2️⃣ Enter traffic information")
    st.write("3️⃣ Click Predict")
    st.write("4️⃣ AI analyzes the inputs")
    st.write("5️⃣ View predicted severity")

    st.divider()

    st.info(
        "This application uses a Random Forest machine learning "
        "model trained on road accident data."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚦 SafeRoad AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Road Accident Severity Prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown("""
<div class="card">

### 🛣️ Analyze Road Accident Risk

Enter the environmental and traffic conditions below.
SafeRoad AI will use the trained machine learning model
to predict the likely accident severity.

</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("🌦️ Environmental Conditions")

col1, col2 = st.columns(2)

with col1:

    weather = st.selectbox(
        "Weather Conditions",
        encoders["Weather_Conditions"].classes_,
        key="weather_conditions"
    )

with col2:

    road_surface = st.selectbox(
        "Road Surface Conditions",
        encoders["Road_Surface_Conditions"].classes_,
        key="road_surface_conditions"
    )


col3, col4 = st.columns(2)

with col3:

    light = st.selectbox(
        "Light Conditions",
        encoders["Light_Conditions"].classes_,
        key="light_conditions"
    )

with col4:

    road_type = st.selectbox(
        "Road Type",
        encoders["Road_Type"].classes_,
        key="road_type"
    )


# =========================================================
# TRAFFIC INFORMATION
# =========================================================

st.subheader("🚗 Traffic Information")

col5, col6, col7 = st.columns(3)

with col5:

    speed = st.number_input(
        "Speed Limit",
        min_value=20,
        max_value=70,
        value=30,
        step=10,
        key="speed_limit"
    )

with col6:

    vehicles = st.number_input(
        "Number of Vehicles",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        key="number_of_vehicles"
    )

with col7:

    casualties = st.number_input(
        "Number of Casualties",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key="number_of_casualties"
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_col, reset_col = st.columns([3, 1])

with predict_col:

    predict_button = st.button(
        "🚨 Analyze Accident Risk",
        use_container_width=True,
        type="primary",
        key="predict_button"
    )

with reset_col:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True,
        key="reset_button"
    )

if reset_button:

    st.rerun()


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # Create dataframe
        input_df = pd.DataFrame({
            "Weather_Conditions": [weather],
            "Road_Surface_Conditions": [road_surface],
            "Light_Conditions": [light],
            "Road_Type": [road_type],
            "Speed_limit": [speed],
            "Number_of_Vehicles": [vehicles],
            "Number_of_Casualties": [casualties]
        })

        # Encode categorical columns
        categorical_columns = [
            "Weather_Conditions",
            "Road_Surface_Conditions",
            "Light_Conditions",
            "Road_Type"
        ]

        for column in categorical_columns:

            input_df[column] = encoders[column].transform(
                input_df[column]
            )

        # Predict
        prediction = model.predict(input_df)

        # Prediction probabilities
        probabilities = model.predict_proba(input_df)[0]

        predicted_class = int(prediction[0])

        severity = {
            0: "Fatal",
            1: "Serious",
            2: "Slight"
        }

        predicted_severity = severity.get(
            predicted_class,
            "Unknown"
        )

        # Find probability of predicted class
        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        confidence = class_probabilities.get(
            predicted_class,
            0
        ) * 100


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("🎯 AI Prediction")

        if predicted_severity == "Fatal":

            st.error(
                "🔴 HIGH RISK — Predicted Severity: FATAL"
            )

        elif predicted_severity == "Serious":

            st.warning(
                "🟠 MEDIUM-HIGH RISK — Predicted Severity: SERIOUS"
            )

        else:

            st.success(
                "🟢 LOWER RISK — Predicted Severity: SLIGHT"
            )


        # Result columns

        result1, result2, result3 = st.columns(3)

        with result1:

            st.metric(
                "Predicted Severity",
                predicted_severity
            )

        with result2:

            st.metric(
                "Model Confidence",
                f"{confidence:.1f}%"
            )

        with result3:

            st.metric(
                "Vehicles",
                vehicles
            )


        # =================================================
        # PROBABILITY CHART
        # =================================================

        st.subheader("📊 Prediction Probability")

        probability_df = pd.DataFrame({
            "Severity": [
                severity.get(int(cls), str(cls))
                for cls in model.classes_
            ],
            "Probability": [
                round(prob * 100, 2)
                for prob in probabilities
            ]
        })

        st.bar_chart(
            probability_df.set_index("Severity")
        )


        # =================================================
        # INPUT SUMMARY
        # =================================================

        st.subheader("📋 Accident Conditions")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write("🌦️ **Weather:**", weather)
            st.write("🛣️ **Road Surface:**", road_surface)
            st.write("💡 **Lighting:**", light)
            st.write("🛤️ **Road Type:**", road_type)

        with summary_col2:

            st.write("⚡ **Speed Limit:**", speed)
            st.write("🚗 **Vehicles:**", vehicles)
            st.write("👥 **Casualties:**", casualties)


        # =================================================
        # SIMPLE EXPLANATION
        # =================================================

        st.subheader("💡 Risk Factors")

        factors = []

        if speed >= 50:
            factors.append("⚡ Higher speed limit")

        if vehicles >= 5:
            factors.append("🚗 High number of vehicles")

        if casualties >= 3:
            factors.append("👥 Multiple casualties")

        if "Rain" in weather or "Snow" in weather:
            factors.append("🌧️ Difficult weather conditions")

        if "Wet" in road_surface or "Snow" in road_surface:
            factors.append("💧 Difficult road surface")

        if "Darkness" in light:
            factors.append("🌙 Low-light conditions")

        if factors:

            for factor in factors:
                st.write("•", factor)

        else:

            st.write(
                "✅ No major risk factors detected from the "
                "selected inputs."
            )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )