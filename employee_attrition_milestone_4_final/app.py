from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from preprocessing import get_risk_level, prepare_input_for_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_attrition_model.pkl"
THRESHOLD_PATH = BASE_DIR / "selected_threshold.txt"
LOGS_PATH = BASE_DIR / "prediction_logs.csv"

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    threshold = float(THRESHOLD_PATH.read_text().strip())
    return model, threshold


def append_prediction_log(raw_input: dict, probability: float, threshold: float, predicted_class: int, risk_level: str) -> None:
    log_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "v1_lightgbm_milestone3",
        "predicted_probability": round(float(probability), 6),
        "selected_threshold": float(threshold),
        "predicted_class": int(predicted_class),
        "risk_level": risk_level,
    }
    log_row.update(raw_input)
    pd.DataFrame([log_row]).to_csv(
        LOGS_PATH,
        mode="a",
        header=not LOGS_PATH.exists(),
        index=False,
    )


def prediction_form() -> dict | None:
    st.subheader("Enter Employee/Candidate Data")
    with st.form("employee_attrition_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            city_development_index = st.number_input("City Development Index", 0.0, 1.0, 0.920, step=0.001)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
            relevent_experience = st.selectbox(
                "Relevant Experience",
                ["Has relevent experience", "No relevent experience"],
            )
            enrolled_university = st.selectbox(
                "Enrolled University",
                ["no_enrollment", "Part time course", "Full time course", "Unknown"],
            )
        with col2:
            education_level = st.selectbox(
                "Education Level",
                ["Graduate", "Masters", "Phd", "High School", "Primary School", "Unknown"],
            )
            major_discipline = st.selectbox(
                "Major Discipline",
                ["STEM", "Humanities", "Business Degree", "Arts", "No Major", "Other", "Unknown"],
            )
            experience = st.selectbox(
                "Experience",
                ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20", "Unknown"],
                index=10,
            )
            last_new_job = st.selectbox("Last New Job", ["never", "1", "2", "3", "4", ">4", "Unknown"])
        with col3:
            company_size = st.selectbox(
                "Company Size",
                ["<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+", "Unknown"],
                index=3,
            )
            company_type = st.selectbox(
                "Company Type",
                ["Pvt Ltd", "Funded Startup", "Early Stage Startup", "Public Sector", "NGO", "Other", "Unknown"],
            )
            training_hours = st.number_input("Training Hours", 1, 400, 36, step=1)

        submitted = st.form_submit_button("Predict Attrition Risk")

    if not submitted:
        return None

    return {
        "city_development_index": float(city_development_index),
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "last_new_job": last_new_job,
        "training_hours": float(training_hours),
    }


def show_monitoring_dashboard() -> None:
    st.subheader("Deployment Monitoring Dashboard")
    if not LOGS_PATH.exists() or LOGS_PATH.stat().st_size == 0:
        st.info("No predictions have been logged yet.")
        return

    logs = pd.read_csv(LOGS_PATH)
    if logs.empty:
        st.info("Prediction log exists but is empty.")
        return

    logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")

    total_predictions = len(logs)
    avg_probability = logs["predicted_probability"].mean()
    high_risk_pct = (logs["risk_level"] == "High Risk").mean() * 100
    class_1_pct = (logs["predicted_class"] == 1).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", f"{total_predictions:,}")
    c2.metric("Average Probability", f"{avg_probability:.2%}")
    c3.metric("High Risk %", f"{high_risk_pct:.1f}%")
    c4.metric("Predicted Class 1 %", f"{class_1_pct:.1f}%")

    if high_risk_pct > 50:
        st.warning("Alert: high-risk predictions exceed 50% of logged predictions.")
    if avg_probability > 0.70:
        st.warning("Alert: average predicted probability is above 70%.")

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = logs["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        st.plotly_chart(px.bar(risk_counts, x="risk_level", y="count", title="Risk Level Distribution"), use_container_width=True)
    with col2:
        valid_ts = logs.dropna(subset=["timestamp"]).copy()
        if not valid_ts.empty:
            daily = valid_ts.set_index("timestamp").resample("D")["predicted_probability"].mean().reset_index()
            st.plotly_chart(px.line(daily, x="timestamp", y="predicted_probability", title="Average Probability Over Time"), use_container_width=True)

    st.write("Recent predictions")
    st.dataframe(logs.tail(20), use_container_width=True)


model, threshold = load_assets()

st.title("Employee Attrition / Job Change Risk Prediction")
st.caption("Milestone 4: Streamlit deployment using the saved full preprocessing pipeline and final LightGBM model.")

tab_predict, tab_monitoring = st.tabs(["Predict", "Monitoring"])

with tab_predict:
    raw_input = prediction_form()
    if raw_input is not None:
        try:
            processed_df = prepare_input_for_model(raw_input)
            probability = float(model.predict_proba(processed_df)[0][1])
            predicted_class = int(probability >= threshold)
            risk_level = get_risk_level(probability)
            append_prediction_log(raw_input, probability, threshold, predicted_class, risk_level)

            st.divider()
            st.subheader("Prediction Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Attrition Probability", f"{probability:.2%}")
            r2.metric("Selected Threshold", f"{threshold:.2f}")
            r3.metric("Predicted Class", predicted_class)

            if risk_level == "High Risk":
                st.error(f"Risk Level: {risk_level}")
            elif risk_level == "Medium Risk":
                st.warning(f"Risk Level: {risk_level}")
            else:
                st.success(f"Risk Level: {risk_level}")
        except Exception as exc:
            st.exception(exc)

with tab_monitoring:
    show_monitoring_dashboard()
