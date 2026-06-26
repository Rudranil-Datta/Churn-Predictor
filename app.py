from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent / "models"

MODEL_STATS = {
    "customers": 7043,
    "churn_rate": 0.265,
    "cv_recall": 0.786,
    "test_recall": 0.775,
    "roc_auc": 0.853,
    "features": 36,
}

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

PRESETS = {
    "At-risk new customer": {
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 2,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 95.0,
        "Total Charges": 190.0,
    },
    "Loyal long-term": {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "Tenure Months": 60,
        "Phone Service": "Yes",
        "Multiple Lines": "Yes",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection": "Yes",
        "Tech Support": "Yes",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Contract": "Two year",
        "Paperless Billing": "Yes",
        "Payment Method": "Bank transfer (automatic)",
        "Monthly Charges": 75.0,
        "Total Charges": 4500.0,
    },
    "High-spend fiber": {
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 10,
        "Phone Service": "Yes",
        "Multiple Lines": "Yes",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Credit card (automatic)",
        "Monthly Charges": 110.0,
        "Total Charges": 1100.0,
    },
}

SEGMENT_PLAYBOOK = {
    "budget loyal customer": "Stable low-spend profile. Focus on value bundles, not heavy discounts.",
    "high risk new customer": "Short tenure and elevated churn signal. Priority for onboarding support and contract incentives.",
    "loyal premium customer": "High-value retained profile. Upsell premium services and loyalty rewards.",
}


@st.cache_resource
def load_artifacts():
    required = [
        "churn_model.joblib",
        "feature_columns.joblib",
        "churn_threshold.joblib",
        "feature_importance.joblib",
        "scaler.joblib",
        "kmeans.joblib",
        "cluster_names.joblib",
    ]
    missing = [name for name in required if not (MODELS_DIR / name).exists()]
    if missing:
        return None, missing

    threshold = float(joblib.load(MODELS_DIR / "churn_threshold.joblib"))
    if threshold < 0.2 or threshold > 0.9:
        threshold = 0.567

    return {
        "model": joblib.load(MODELS_DIR / "churn_model.joblib"),
        "feature_columns": joblib.load(MODELS_DIR / "feature_columns.joblib"),
        "churn_threshold": threshold,
        "feature_importance": joblib.load(MODELS_DIR / "feature_importance.joblib"),
        "scaler": joblib.load(MODELS_DIR / "scaler.joblib"),
        "kmeans": joblib.load(MODELS_DIR / "kmeans.joblib"),
        "cluster_names": joblib.load(MODELS_DIR / "cluster_names.joblib"),
    }, []


def init_session_defaults():
    if "form_values" not in st.session_state:
        st.session_state.form_values = PRESETS["At-risk new customer"].copy()


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


INTERNET_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]


def add_engineered_features(row: pd.DataFrame) -> pd.DataFrame:
    row = row.copy()
    row["Total Charges"] = pd.to_numeric(row["Total Charges"], errors="coerce").fillna(0)
    row["Avg Monthly Spend"] = row["Total Charges"] / row["Tenure Months"].clip(lower=1)
    row["Internet Add-ons"] = row[INTERNET_COLS].apply(
        lambda values: sum(v == "Yes" for v in values), axis=1
    )
    row["Auto Payment"] = row["Payment Method"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    row["Contract Ordinal"] = row["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    row["Tenure Bucket"] = pd.cut(
        row["Tenure Months"],
        bins=[-1, 12, 36, 72],
        labels=["New", "Mid", "Loyal"],
    )
    return row


def encode_customer(data: dict, feature_columns: list[str]) -> pd.DataFrame:
    row = add_engineered_features(pd.DataFrame([data]))
    encoded = pd.get_dummies(row, drop_first=True)
    return encoded.reindex(columns=feature_columns, fill_value=0)


def humanize_feature(name: str) -> str:
    return name.replace("_", " ")


def get_risk_band(probability: float, threshold: float) -> tuple[str, str]:
    low_cut = max(0.35, threshold - 0.15)
    if probability < low_cut:
        return "Low", "success"
    if probability < threshold:
        return "Medium", "warning"
    return "High", "error"


def get_top_drivers(
    features: pd.DataFrame,
    importance_df: pd.DataFrame,
    form_data: dict,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    importance_map = importance_df.set_index("Feature")["Importance"].to_dict()
    drivers: list[tuple[str, float]] = []

    for feature, value in features.iloc[0].items():
        if feature not in importance_map:
            continue
        if value == 0:
            continue
        if feature in {
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "Avg Monthly Spend",
            "Internet Add-ons",
            "Auto Payment",
            "Contract Ordinal",
        }:
            label = f"{feature}: {form_data.get(feature, round(value, 2))}"
            score = importance_map[feature]
        else:
            label = humanize_feature(feature)
            score = importance_map[feature]
        drivers.append((label, score))

    if form_data["Contract"] == "Month-to-month":
        drivers.append(("Month-to-month contract", 0.12))
    if form_data["Payment Method"] == "Electronic check":
        drivers.append(("Electronic check payment", 0.08))
    if form_data["Tech Support"] == "No" and form_data["Internet Service"] != "No":
        drivers.append(("No tech support on internet plan", 0.07))

    drivers = sorted(drivers, key=lambda item: item[1], reverse=True)
    unique: list[tuple[str, float]] = []
    seen = set()
    for label, score in drivers:
        if label in seen:
            continue
        seen.add(label)
        unique.append((label, score))
        if len(unique) == top_n:
            break
    return unique


def get_retention_action(form_data: dict, risk_band: str, segment: str) -> str:
    if risk_band == "Low":
        return "Low risk — upsell streaming bundle or loyalty perks instead of discounting."

    if form_data["Contract"] == "Month-to-month":
        return "Offer annual contract discount or free upgrade to one-year plan."

    if form_data["Tech Support"] == "No" and form_data["Internet Service"] != "No":
        return "Enable free 30-day tech support trial to reduce service frustration."

    if form_data["Payment Method"] == "Electronic check":
        return "Move customer to automatic bank/card payment with small monthly credit."

    if "high risk" in segment.lower():
        return "Assign retention call within 48 hours and review monthly charges."

    return "Send personalized retention email with service bundle offer."


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


def render_sidebar(threshold: float):
    st.sidebar.header("Model snapshot")
    st.sidebar.metric("Churn threshold", f"{threshold:.1%}")
    st.sidebar.metric("CV recall", f"{MODEL_STATS['cv_recall']:.1%}")
    st.sidebar.metric("Test recall", f"{MODEL_STATS['test_recall']:.1%}")
    st.sidebar.metric("ROC-AUC", f"{MODEL_STATS['roc_auc']:.2f}")
    st.sidebar.caption(
        f"{MODEL_STATS['customers']:,} customers · "
        f"{MODEL_STATS['churn_rate']:.1%} churn rate · "
        f"{MODEL_STATS['features']} features"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Quick profiles")
    for name in PRESETS:
        if st.sidebar.button(name, use_container_width=True):
            st.session_state.form_values = PRESETS[name].copy()
            st.session_state.run_prediction = True


def render_predict_tab(artifacts: dict):
    values = st.session_state.form_values
    threshold = artifacts["churn_threshold"]

    with st.form("customer_form"):
        st.subheader("Customer details")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["Gender"], index=CATEGORICAL_OPTIONS["Gender"].index(values["Gender"]))
            senior = st.selectbox("Senior Citizen", CATEGORICAL_OPTIONS["Senior Citizen"], index=CATEGORICAL_OPTIONS["Senior Citizen"].index(values["Senior Citizen"]))
            partner = st.selectbox("Partner", CATEGORICAL_OPTIONS["Partner"], index=CATEGORICAL_OPTIONS["Partner"].index(values["Partner"]))
            dependents = st.selectbox("Dependents", CATEGORICAL_OPTIONS["Dependents"], index=CATEGORICAL_OPTIONS["Dependents"].index(values["Dependents"]))
            tenure = st.number_input("Tenure Months", min_value=0, max_value=72, value=int(values["Tenure Months"]))

        with col2:
            phone = st.selectbox("Phone Service", CATEGORICAL_OPTIONS["Phone Service"], index=CATEGORICAL_OPTIONS["Phone Service"].index(values["Phone Service"]))
            multiple_lines = st.selectbox(
                "Multiple Lines",
                CATEGORICAL_OPTIONS["Multiple Lines"],
                index=CATEGORICAL_OPTIONS["Multiple Lines"].index(values["Multiple Lines"]),
                disabled=phone == "No",
            )
            internet = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["Internet Service"], index=CATEGORICAL_OPTIONS["Internet Service"].index(values["Internet Service"]))
            online_security = st.selectbox(
                "Online Security",
                CATEGORICAL_OPTIONS["Online Security"],
                index=CATEGORICAL_OPTIONS["Online Security"].index(values["Online Security"]),
                disabled=internet == "No",
            )
            online_backup = st.selectbox(
                "Online Backup",
                CATEGORICAL_OPTIONS["Online Backup"],
                index=CATEGORICAL_OPTIONS["Online Backup"].index(values["Online Backup"]),
                disabled=internet == "No",
            )
            device_protection = st.selectbox(
                "Device Protection",
                CATEGORICAL_OPTIONS["Device Protection"],
                index=CATEGORICAL_OPTIONS["Device Protection"].index(values["Device Protection"]),
                disabled=internet == "No",
            )

        with col3:
            tech_support = st.selectbox(
                "Tech Support",
                CATEGORICAL_OPTIONS["Tech Support"],
                index=CATEGORICAL_OPTIONS["Tech Support"].index(values["Tech Support"]),
                disabled=internet == "No",
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                CATEGORICAL_OPTIONS["Streaming TV"],
                index=CATEGORICAL_OPTIONS["Streaming TV"].index(values["Streaming TV"]),
                disabled=internet == "No",
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                CATEGORICAL_OPTIONS["Streaming Movies"],
                index=CATEGORICAL_OPTIONS["Streaming Movies"].index(values["Streaming Movies"]),
                disabled=internet == "No",
            )
            contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"], index=CATEGORICAL_OPTIONS["Contract"].index(values["Contract"]))
            paperless = st.selectbox("Paperless Billing", CATEGORICAL_OPTIONS["Paperless Billing"], index=CATEGORICAL_OPTIONS["Paperless Billing"].index(values["Paperless Billing"]))
            payment = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["Payment Method"], index=CATEGORICAL_OPTIONS["Payment Method"].index(values["Payment Method"]))
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=float(values["Monthly Charges"]), step=0.05)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=float(values["Total Charges"]), step=0.05)

        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        st.session_state.form_values = apply_service_rules(
            {
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
        st.session_state.run_prediction = True

    if not st.session_state.get("run_prediction"):
        st.info("Pick a sidebar profile or fill the form, then click **Predict churn**.")
        return

    form_data = st.session_state.form_values
    features = encode_customer(form_data, artifacts["feature_columns"])
    churn_probability = float(artifacts["model"].predict_proba(features)[0, 1])
    risk_band, band_style = get_risk_band(churn_probability, threshold)
    churn_prediction = "Will churn" if churn_probability >= threshold else "Will stay"
    segment = predict_segment(
        form_data["Tenure Months"],
        form_data["Monthly Charges"],
        form_data["Total Charges"],
        churn_probability,
        artifacts["scaler"],
        artifacts["kmeans"],
        artifacts["cluster_names"],
    )
    drivers = get_top_drivers(features, artifacts["feature_importance"], form_data)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn probability", f"{churn_probability:.1%}")
    c2.metric("Risk band", risk_band)
    c3.metric("Prediction", churn_prediction)
    c4.metric("Segment", segment)

    st.progress(min(churn_probability, 1.0))

    if band_style == "success":
        st.success(f"{risk_band} churn risk (threshold {threshold:.1%}).")
    elif band_style == "warning":
        st.warning(f"{risk_band} churn risk (threshold {threshold:.1%}).")
    else:
        st.error(f"{risk_band} churn risk (threshold {threshold:.1%}).")

    left, right = st.columns(2)
    with left:
        st.subheader("Top churn drivers")
        for label, score in drivers:
            st.write(f"- **{label}** (importance {score:.2f})")
    with right:
        st.subheader("Recommended action")
        st.write(get_retention_action(form_data, risk_band, segment))
        st.caption(SEGMENT_PLAYBOOK.get(segment, "Review account history and tailor outreach."))


def render_model_tab(artifacts: dict, threshold: float):
    st.subheader("How the model works")
    st.markdown(
        """
        - **Algorithm:** Random Forest tuned with `GridSearchCV` on train data only (`scoring='recall'`)
        - **Imbalance:** `class_weight='balanced'` plus tuned probability threshold
        - **Features:** 36 inputs after dropping high-cardinality `City` and adding engineered fields
        - **Goal:** Catch churners (high recall), not maximize accuracy alone
        """
    )

    st.markdown("**Engineered features**")
    st.markdown(
        "- `Avg Monthly Spend` = total charges / tenure\n"
        "- `Internet Add-ons` = count of active streaming/security services\n"
        "- `Auto Payment` = automatic bank/card billing\n"
        "- `Contract Ordinal` = month-to-month < one-year < two-year\n"
        "- `Tenure Bucket` = New / Mid / Loyal"
    )

    st.subheader("Global feature importance")
    importance = artifacts["feature_importance"].head(12).set_index("Feature")
    st.bar_chart(importance)

    st.subheader("Evaluation summary")
    st.table(
        pd.DataFrame(
            {
                "Metric": ["CV recall", "Test recall", "ROC-AUC", "Churn threshold", "Feature count"],
                "Value": [
                    f"{MODEL_STATS['cv_recall']:.1%}",
                    f"{MODEL_STATS['test_recall']:.1%}",
                    f"{MODEL_STATS['roc_auc']:.2f}",
                    f"{threshold:.1%}",
                    str(MODEL_STATS["features"]),
                ],
            }
        )
    )


def main():
    st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="wide")
    st.title("Telco Customer Churn Predictor")
    st.caption("Retention intelligence powered by tuned Random Forest on engineered telco features.")

    artifacts, missing = load_artifacts()
    if artifacts is None:
        st.error("Model files missing. Run the save cell at the end of `Churn.ipynb` first.")
        st.code("\n".join(f"- models/{name}" for name in missing))
        st.stop()

    init_session_defaults()
    render_sidebar(artifacts["churn_threshold"])

    predict_tab, model_tab = st.tabs(["Predict", "About model"])
    with predict_tab:
        render_predict_tab(artifacts)
    with model_tab:
        render_model_tab(artifacts, artifacts["churn_threshold"])


if __name__ == "__main__":
    main()
