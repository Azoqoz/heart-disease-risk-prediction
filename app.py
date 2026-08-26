import streamlit as st
import pandas as pd
import joblib


# Load the audited model artifacts.
model = joblib.load("heart_model.joblib")
scaler = joblib.load("scaler.joblib")


st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --background: #f3f0e9;
        --surface: #faf8f3;
        --surface-secondary: #ece8df;
        --rule: #d5d0c6;
        --rule-strong: #c5bfb4;
        --text: #17191b;
        --text-soft: #3f403f;
        --text-muted: #68645d;
        --action: #202326;
        --burgundy: #8e3b3b;
        --green: #557368;
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1540px;
        padding: 2.8rem 2.5rem 3.25rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    footer {
        display: none;
    }

    .report-header {
        padding-bottom: 1.6rem;
        border-bottom: 1px solid var(--rule);
    }

    .report-header h1 {
        max-width: 900px;
        margin: 0;
        color: var(--text);
        font-size: clamp(2.2rem, 3.6vw, 3.15rem);
        font-weight: 600;
        line-height: 1.1;
        letter-spacing: -0.04em;
    }

    .report-header p {
        max-width: 740px;
        margin: 0.65rem 0 0;
        color: var(--text-muted);
        font-size: 1.03rem;
        line-height: 1.6;
    }

    .safety-note {
        margin: 0.9rem 0 1.8rem;
        padding: 0.68rem 0.9rem;
        border-left: 3px solid var(--rule-strong);
        background: rgba(236, 232, 223, 0.62);
        color: #5c5953;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    [data-testid="stHorizontalBlock"]:has(.workspace-heading) {
        align-items: flex-start;
        gap: clamp(3.5rem, 5vw, 5.75rem);
    }

    [data-testid="stHorizontalBlock"]:has(.workspace-heading) > [data-testid="stColumn"] {
        align-self: flex-start;
        padding-top: 0;
    }

    .workspace-heading {
        margin: 0 0 0.25rem;
        color: var(--text);
        font-size: 1.45rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    .workspace-note {
        margin: 0 0 1.35rem;
        color: var(--text-muted);
        font-size: 0.93rem;
        line-height: 1.5;
    }

    .form-section {
        margin: 0.1rem 0 0.85rem;
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 600;
    }

    .form-rule {
        height: 1px;
        margin: 0.75rem 0 1.2rem;
        background: var(--rule);
    }

    [data-testid="stWidgetLabel"] p {
        color: #4f4d48;
        font-size: 0.86rem;
        font-weight: 540;
        letter-spacing: 0;
    }

    [data-testid="stNumberInput"] [data-baseweb="input"],
    div[data-baseweb="select"] > div {
        min-height: 2.75rem;
        border: 1px solid var(--rule) !important;
        border-radius: 4px !important;
        background: var(--surface) !important;
        color: var(--text) !important;
        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] [data-baseweb="base-input"],
    [data-testid="stNumberInput"] input {
        background: var(--surface) !important;
        color: var(--text) !important;
    }

    [data-testid="stNumberInput"] input {
        min-height: 2.65rem;
        border: 0 !important;
        font-size: 0.88rem;
        -webkit-text-fill-color: var(--text) !important;
    }

    [data-testid="stNumberInput"] button {
        min-width: 2.7rem;
        border: 0 !important;
        border-left: 1px solid var(--rule) !important;
        border-radius: 0 !important;
        background: var(--surface-secondary) !important;
        color: var(--text) !important;
        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background: #e3ded4 !important;
        color: var(--text) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        color: var(--text) !important;
        font-size: 0.88rem;
    }

    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stSelectbox"] div:has(> [role="combobox"]) {
        background: var(--surface) !important;
        color: var(--text) !important;
    }

    [data-testid="stSelectbox"] div:has(> [role="combobox"]) {
        min-height: 2.75rem;
        border: 1px solid var(--rule) !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }

    [data-testid="stSelectbox"] button[aria-label="Open"] {
        background: transparent !important;
        color: var(--text-muted) !important;
    }

    div[data-baseweb="select"] svg {
        fill: var(--text-muted) !important;
        color: var(--text-muted) !important;
    }

    [data-testid="stNumberInput"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #8f897f !important;
        box-shadow: 0 0 0 1px #8f897f !important;
    }

    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] ul,
    [role="listbox"] {
        border-color: var(--rule) !important;
        border-radius: 4px !important;
        background: var(--surface) !important;
        color: var(--text) !important;
        box-shadow: 0 8px 24px rgba(32, 35, 38, 0.12) !important;
    }

    [role="option"] {
        background: var(--surface) !important;
        color: var(--text) !important;
        font-size: 0.88rem;
    }

    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background: var(--surface-secondary) !important;
        color: var(--text) !important;
    }

    [data-testid="stTooltipIcon"] {
        color: var(--text-muted) !important;
        background: transparent !important;
    }

    [data-testid="stTooltipIcon"] svg {
        fill: var(--text-muted) !important;
        color: var(--text-muted) !important;
    }

    button[aria-label^="Help for"] {
        border: 0 !important;
        background: transparent !important;
        color: var(--text-muted) !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"] {
        border: 0 !important;
        border-top: 1px solid var(--rule) !important;
        border-bottom: 1px solid var(--rule) !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"]:focus-within,
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details:focus-within {
        border-color: var(--rule) !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"] details {
        border: 0 !important;
        border-radius: 0 !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details[open] > summary,
    [data-testid="stExpander"] summary[aria-expanded="true"] {
        color: var(--text-soft) !important;
        background: transparent !important;
        font-size: 0.88rem;
        font-weight: 540;
    }

    [data-testid="stExpander"] summary:hover {
        color: var(--text) !important;
        background: var(--surface-secondary) !important;
    }

    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:focus-visible {
        border-color: transparent !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] summary span {
        color: var(--text-muted) !important;
        fill: var(--text-muted) !important;
        background: transparent !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--text-muted) !important;
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .stButton {
        margin: 1.35rem 0;
    }

    .stButton > button {
        min-height: 2.75rem;
        margin-top: 0;
        border: 1px solid var(--action);
        border-radius: 4px;
        background: var(--action);
        color: #f7f3eb;
        font-size: 0.88rem;
        font-weight: 600;
        box-shadow: none;
        transition: background-color 120ms ease, border-color 120ms ease;
    }

    .stButton > button:hover,
    .stButton > button:focus:not(:active) {
        border-color: #33373a;
        background: #33373a;
        color: #f7f3eb;
        box-shadow: none;
    }

    .result-empty {
        min-height: 230px;
        padding-top: 0.25rem;
    }

    .result-empty h3 {
        margin: 0;
        color: var(--text-soft);
        font-size: 1.05rem;
        font-weight: 600;
    }

    .result-empty p {
        max-width: 380px;
        margin: 0.55rem 0 0;
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .result-output {
        min-height: 300px;
        padding-top: 0.1rem;
    }

    .result-label {
        margin: 0;
        color: var(--text-muted);
        font-size: 0.86rem;
        font-weight: 540;
    }

    .result-value {
        margin: 0.6rem 0 0;
        color: var(--text);
        font-size: clamp(3.5rem, 8vw, 5.8rem);
        font-weight: 600;
        line-height: 0.98;
        letter-spacing: -0.06em;
    }

    .result-classification {
        margin: 0.85rem 0 0;
        font-size: 1rem;
        font-weight: 600;
    }

    .result-value.lower {
        color: var(--green);
    }

    .result-value.higher {
        color: var(--burgundy);
    }

    .result-classification.lower {
        color: var(--green);
    }

    .result-classification.higher {
        color: var(--burgundy);
    }

    .probability-scale {
        position: relative;
        height: 1px;
        margin: 2.25rem 0 0;
        background: var(--rule-strong);
    }

    .probability-marker,
    .threshold-marker {
        position: absolute;
        top: -4px;
        width: 1px;
        height: 9px;
    }

    .probability-marker.lower {
        background: var(--green);
    }

    .probability-marker.higher {
        background: var(--burgundy);
    }

    .threshold-marker {
        left: 50%;
        background: #777269;
    }

    .probability-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.55rem;
        color: var(--text-muted);
        font-size: 0.72rem;
    }

    .result-explanation {
        max-width: 500px;
        margin: 1.45rem 0 0;
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .technical-details {
        margin-top: 1.1rem;
        border-top: 1px solid var(--rule);
    }

    .technical-row {
        display: grid;
        grid-template-columns: 92px 1fr;
        gap: 1rem;
        padding: 0.48rem 0;
        border-bottom: 1px solid var(--rule);
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.4;
    }

    .technical-row span:last-child {
        color: var(--text-soft);
    }

    .result-disclaimer {
        max-width: 500px;
        margin: 0.9rem 0 0;
        color: var(--text-muted);
        font-size: 0.85rem;
        line-height: 1.55;
    }

    @media (max-width: 760px) {
        .block-container {
            padding: 1.6rem 1rem 2.5rem;
        }

        .report-header h1 {
            font-size: 2rem;
        }

        .result-empty,
        .result-output {
            min-height: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


def render_result(risk_probability):
    if risk_probability is None:
        st.markdown(
            """
            <div class="result-empty">
                <h3>No estimate generated</h3>
                <p>Complete the patient data and select Estimate.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    is_higher_probability = risk_probability >= 0.50
    classification = "Higher probability" if is_higher_probability else "Lower probability"
    classification_class = "higher" if is_higher_probability else "lower"
    explanation = (
        "The model estimates a higher probability of the heart-disease class for the provided inputs."
        if is_higher_probability
        else "The model estimates a lower probability of the heart-disease class for the provided inputs."
    )
    risk_percent = risk_probability * 100

    st.markdown(
        f"""
        <div class="result-output">
            <p class="result-label">Estimated Heart Disease Probability</p>
            <div class="result-value {classification_class}">{risk_percent:.1f}%</div>
            <p class="result-classification {classification_class}">{classification}</p>
            <div class="probability-scale" aria-label="Estimated probability {risk_percent:.1f} percent">
                <span class="threshold-marker"></span>
                <span class="probability-marker {classification_class}" style="left: {risk_percent:.2f}%"></span>
            </div>
            <div class="probability-labels"><span>0%</span><span>50% threshold</span><span>100%</span></div>
            <p class="result-explanation">{explanation}</p>
            <div class="technical-details">
                <div class="technical-row"><span>Threshold</span><span>50%</span></div>
                <div class="technical-row"><span>Model</span><span>XGBoost</span></div>
                <div class="technical-row"><span>Features</span><span>13</span></div>
            </div>
            <p class="result-disclaimer">
                This result is a machine-learning estimate for educational purposes and is not a medical diagnosis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if "risk_probability" not in st.session_state:
    st.session_state.risk_probability = None


st.markdown(
    """
    <header class="report-header">
        <h1>Heart Disease Risk Prediction</h1>
        <p>Clinical input-based machine learning screening tool for educational risk estimation.</p>
    </header>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="safety-note">
        Educational screening tool only. This application estimates model-based risk from the provided clinical
        inputs and does not provide a medical diagnosis or replace evaluation by a qualified healthcare professional.
    </p>
    """,
    unsafe_allow_html=True
)


input_column, result_column = st.columns([3, 2], gap="large")

with input_column:
    st.markdown('<h2 class="workspace-heading">Patient Data</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="workspace-note">Enter the available patient and clinical measurements.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<h3 class="form-section">Patient</h3>', unsafe_allow_html=True)
    patient_left, patient_right = st.columns(2, gap="medium")

    with patient_left:
        age = st.number_input("Age", 1, 100, 25, help="Age in years.")

    with patient_right:
        sex_option = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex_option == "Male" else 0

    st.markdown('<div class="form-rule"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="form-section">Measurements</h3>', unsafe_allow_html=True)
    measurement_left, measurement_right = st.columns(2, gap="medium")

    with measurement_left:
        trestbps = st.number_input(
            "Resting Blood Pressure",
            50,
            250,
            120,
            help="Measured in mmHg."
        )

    with measurement_right:
        chol = st.number_input(
            "Cholesterol Level",
            100,
            600,
            200,
            help="Measured in mg/dL."
        )

    measurement_lower, measurement_spacer = st.columns(2, gap="medium")

    with measurement_lower:
        thalach = st.number_input(
            "Maximum Heart Rate",
            50,
            250,
            150,
            help="Measured in beats per minute (bpm)."
        )

    st.markdown('<div class="form-rule"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="form-section">Symptoms</h3>', unsafe_allow_html=True)

    cp_option = st.selectbox(
        "Chest Pain Type",
        [
            "Chest pain during physical activity",
            "Mild or unusual chest pain",
            "Chest pain not related to the heart",
            "No chest pain symptoms"
        ],
        help="The selected description is converted to the dataset's chest-pain category code."
    )

    cp_mapping = {
        "Chest pain during physical activity": 0,
        "Mild or unusual chest pain": 1,
        "Chest pain not related to the heart": 2,
        "No chest pain symptoms": 3
    }
    cp = cp_mapping[cp_option]

    symptom_left, symptom_right = st.columns(2, gap="medium")

    with symptom_left:
        exang_option = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        exang = 1 if exang_option == "Yes" else 0

    with symptom_right:
        fbs_option = st.selectbox(
            "High Fasting Blood Sugar",
            ["No", "Yes"],
            help="Dataset indicator for fasting blood sugar above its recorded threshold."
        )
        fbs = 1 if fbs_option == "Yes" else 0

    st.markdown('<div class="form-rule"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="form-section">Clinical Indicators</h3>', unsafe_allow_html=True)

    with st.expander("Show clinical indicators"):
        st.caption("These fields correspond to ECG or other clinical test measurements in the dataset.")
        advanced_left, advanced_right = st.columns(2, gap="medium")

        with advanced_left:
            restecg = st.selectbox(
                "Rest ECG",
                [0, 1, 2],
                help="Dataset category code: 0, 1, or 2."
            )
            oldpeak = st.number_input(
                "ST Depression",
                0.0,
                10.0,
                1.0,
                help="ST depression value recorded relative to rest."
            )
            slope = st.selectbox(
                "ST Segment Slope",
                [0, 1, 2],
                help="Dataset category code: 0, 1, or 2."
            )

        with advanced_right:
            ca = st.selectbox(
                "Major Vessels",
                [0, 1, 2, 3, 4],
                help="Dataset value ranging from 0 to 4."
            )
            thal = st.selectbox(
                "Thalassemia Test Result",
                [0, 1, 2, 3],
                help="Dataset category code ranging from 0 to 3."
            )

    predict = st.button("Estimate Risk", use_container_width=True, type="primary")

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
        st.session_state.risk_probability = float(model.predict_proba(input_scaled)[0][1])


with result_column:
    st.markdown('<h2 class="workspace-heading">Result</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="workspace-note">Model estimate from the most recently submitted patient data.</p>',
        unsafe_allow_html=True
    )
    render_result(st.session_state.risk_probability)
