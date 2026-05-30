import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load Model
# -------------------------

model = joblib.load("heart_model.joblib")
scaler = joblib.load("scaler.joblib")

# -------------------------
# Page Settings
# -------------------------

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------
# Custom Styling
# -------------------------

st.markdown("""
<style>
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 0px;
}

.sub-text {
    color: #B8C0CC;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------

st.markdown(
    '<div class="main-title">❤️ Heart Disease Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">AI-powered educational risk prediction system</div>',
    unsafe_allow_html=True
)

st.info(
    "This application is an educational machine learning project and is not a medical diagnosis tool."
)

# -------------------------
# Model Performance
# -------------------------

with st.expander("Model Performance"):
    st.write("Model: XGBoost Classifier")
    st.write("Accuracy: 98.54%")
    st.write("Precision: 100%")
    st.write("Recall: 97.09%")
    st.write("F1 Score: 98.52%")

# -------------------------
# Patient Information
# -------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        1,
        100,
        25
    )

    sex_option = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    sex = 1 if sex_option == "Male" else 0

    trestbps = st.number_input(
        "Blood Pressure",
        50,
        250,
        120
    )

with col2:

    chol = st.number_input(
        "Cholesterol Level",
        100,
        600,
        200
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        50,
        250,
        150
    )

    cp_option = st.selectbox(
        "Chest Pain Type",
        [
            "Chest pain during physical activity",
            "Mild or unusual chest pain",
            "Chest pain not related to the heart",
            "No chest pain symptoms"
        ]
    )

# -------------------------
# Chest Pain Mapping
# -------------------------

cp_mapping = {
    "Chest pain during physical activity": 0,
    "Mild or unusual chest pain": 1,
    "Chest pain not related to the heart": 2,
    "No chest pain symptoms": 3
}

cp = cp_mapping[cp_option]

# Internal logic for model
exang = 1 if cp_option == "Chest pain during physical activity" else 0

# -------------------------
# Advanced Medical Inputs
# -------------------------

with st.expander("Advanced Medical Information"):

    st.caption(
        "These values usually come from ECG or medical tests."
    )

    fbs_option = st.selectbox(
        "High Fasting Blood Sugar?",
        ["No", "Yes"]
    )

    fbs = 1 if fbs_option == "Yes" else 0

    restecg = st.selectbox(
        "Rest ECG Result",
        [0, 1, 2]
    )

    oldpeak = st.number_input(
        "ECG ST Depression Value",
        0.0,
        10.0,
        1.0
    )

    slope = st.selectbox(
        "ST Segment Slope",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia Test Result",
        [0, 1, 2, 3]
    )

# -------------------------
# Predict Button
# -------------------------

predict = st.button(
    "Predict Heart Disease Risk",
    use_container_width=True,
    type="secondary"
)

# -------------------------
# Prediction
# -------------------------

if predict:

    input_data = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    input_scaled = scaler.transform(input_data)

    # In this dataset:
    # Class 0 = Heart Disease Risk
    risk_probability = model.predict_proba(input_scaled)[0][0]

    st.subheader("Prediction Result")

    st.write(
        f"Estimated Risk Probability: **{risk_probability:.2%}**"
    )

    st.progress(
        int(risk_probability * 100)
    )

    if risk_probability >= 0.50:
        st.error(
            f"High Risk — estimated risk probability is {risk_probability:.2%}"
        )
    else:
        st.success(
            f"Low Risk — estimated risk probability is {risk_probability:.2%}"
        )

    st.caption(
        "This percentage represents the model's estimated probability of heart disease risk based on the provided inputs."
    )