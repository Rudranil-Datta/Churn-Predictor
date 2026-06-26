from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent / "models"

CATEGORICAL_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone Service": ["No", "Yes"],
    "Multiple Lines": ["No", "No phone service", "Yes"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No", "No internet service", "Yes"],
    "Online Backup": ["No", "No internet service", "Yes"],
    "Device Protection": ["No", "No internet service", "Yes"],
    "Tech Support": ["No", "No internet service", "Yes"],
    "Streaming TV": ["No", "No internet service", "Yes"],
    "Streaming Movies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["No", "Yes"],
    "Payment Method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


@st.cache_resource
def load_artifacts():
    required = [
        "churn_model.joblib",
        "feature_columns.joblib",
        "cities.joblib",
        "scaler.joblib",
        "kmeans.joblib",
        "cluster_names.joblib",
    ]
    missing = [name for name in required if not (MODELS_DIR / name).exists()]
    if missing:
        return None, missing

    return {
        "model": joblib.load(MODELS_DIR / "churn_model.joblib"),
        "feature_columns": joblib.load(MODELS_DIR / "feature_columns.joblib"),
        "cities": joblib.load(MODELS_DIR / "cities.joblib"),
        "scaler": joblib.load(MODELS_DIR / "scaler.joblib"),
        "kmeans": joblib.load(MODELS_DIR / "kmeans.joblib"),
        "cluster_names": joblib.load(MODELS_DIR / "cluster_names.joblib"),
    }, []


def apply_service_rules(form_data: dict) -> dict:
    data = form_data.copy()
    if data["Phone Service"] == "No":
        data["Multiple Lines"] = "No phone service"
    if data["Internet Service"] == "No":
        for field in [
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
        ]:
            data[field] = "No internet service"
    return data


def encode_customer(data: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = pd.DataFrame([data])
    row["Total Charges"] = pd.to_numeric(row["Total Charges"], errors="coerce").fillna(0)
    encoded = pd.get_dummies(row, drop_first=True)
    return encoded.reindex(columns=feature_columns, fill_value=0)


def predict_segment(
    tenure: float,
    monthly_charges: float,
    total_charges: float,
    churn_probability: float,
    scaler,
    kmeans,
    cluster_names: dict,
) -> str:
    segment_input = pd.DataFrame(
        [
            {
                "Tenure Months": tenure,
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges,
                "Churn probability": churn_probability,
            }
        ]
    )
    scaled = scaler.transform(segment_input)
    cluster_id = int(kmeans.predict(scaled)[0])
    return cluster_names.get(cluster_id, f"Cluster {cluster_id}")


def main():
    st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="wide")
    st.title("Telco Customer Churn Predictor")
    st.caption("Predict churn risk and customer segment from service profile.")

    artifacts, missing = load_artifacts()
    if artifacts is None:
        st.error("Model files missing. Run the save cell at the end of `Churn.ipynb` first.")
        st.code("\n".join(f"- models/{name}" for name in missing))
        st.stop()

    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]
    cities = artifacts["cities"]

    with st.form("customer_form"):
        st.subheader("Customer details")
        col1, col2, col3 = st.columns(3)

        with col1:
            city = st.selectbox("City", cities)
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["Gender"])
            senior = st.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["Senior Citizen"])
            partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"])
            dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"])
            tenure = st.number_input("Tenure Months", min_value=0, max_value=72, value=12)

        with col2:
            phone = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["Phone Service"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                CATEGORICAL_OPTIONS["Multiple Lines"],
                disabled=phone == "No",
            )
            internet = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["Internet Service"])
            online_security = st.selectbox(
                "Online Security",
                CATEGORICAL_OPTIONS["Online Security"],
                disabled=internet == "No",
            )
            online_backup = st.selectbox(
                "Online Backup",
                CATEGORICAL_OPTIONS["Online Backup"],
                disabled=internet == "No",
            )
            device_protection = st.selectbox(
                "Device Protection",
                CATEGORICAL_OPTIONS["Device Protection"],
                disabled=internet == "No",
            )

        with col3:
            tech_support = st.selectbox(
                "Tech Support",
                CATEGORICAL_OPTIONS["Tech Support"],
                disabled=internet == "No",
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                CATEGORICAL_OPTIONS["Streaming TV"],
                disabled=internet == "No",
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                CATEGORICAL_OPTIONS["Streaming Movies"],
                disabled=internet == "No",
            )
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
            paperless = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["Paperless Billing"])
            payment = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["Payment Method"])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.05)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0, step=0.05)

        submitted = st.form_submit_button("Predict churn", type="primary")

    if not submitted:
        st.info("Fill customer details and click **Predict churn**.")
        return

    form_data = apply_service_rules(
        {
            "City": city,
            "Gender": gender,
            "Senior Citizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure,
            "Phone Service": phone,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_protection,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges,
        }
    )

    features = encode_customer(form_data, feature_columns)
    churn_probability = float(model.predict_proba(features)[0, 1])
    churn_prediction = "Will churn" if churn_probability >= 0.5 else "Will stay"
    segment = predict_segment(
        tenure,
        monthly_charges,
        total_charges,
        churn_probability,
        artifacts["scaler"],
        artifacts["kmeans"],
        artifacts["cluster_names"],
    )

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Churn probability", f"{churn_probability:.1%}")
    result_col2.metric("Prediction", churn_prediction)
    result_col3.metric("Customer segment", segment)

    st.progress(churn_probability)
    if churn_probability >= 0.5:
        st.warning("High churn risk. Consider retention offer or contract upgrade.")
    else:
        st.success("Low churn risk based on current profile.")


if __name__ == "__main__":
    main()
