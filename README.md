# Telco Customer Churn Predictor

Machine learning project that analyzes telecom customer data, predicts churn risk, and segments customers for retention targeting. Includes a Jupyter notebook for exploration and training, plus a Streamlit app for live predictions with retention recommendations.

## Purpose

Help a telecom provider understand **why customers leave** and **which customers are most likely to churn**, so retention teams can act early. The workflow covers:

1. Exploratory analysis of churn drivers
2. Feature engineering and supervised churn classification with Random Forest
3. Proper hyperparameter tuning with cross-validation (recall-focused)
4. Unsupervised customer segmentation with K-Means
5. A Streamlit retention intelligence app

## Data source

**Dataset:** IBM Telco Customer Churn (`Telco_customer_churn.xlsx`)

- ~7,043 customer records
- Demographics, service subscriptions, billing details, and churn labels
- Target variable: `Churn Value` (0 = stayed, 1 = churned)

Place the Excel file in the project root before running the notebook or generating models.

## Project structure

| File / folder | Role |
|---------------|------|
| `Churn.ipynb` | Full analysis, model training, and model export |
| `app.py` | Streamlit app for churn prediction and segmentation |
| `requirements.txt` | Python dependencies |
| `models/` | Saved artifacts (created by the notebook save cell) |

## Architecture

```text
Telco_customer_churn.xlsx
        │
        ▼
   Churn.ipynb
  clean → EDA → feature engineering → train → tune → save
        │
        ▼
models/*.joblib
        │
        ▼
      app.py
```

## Pipeline stages

### 1. Exploratory data analysis

- Load and inspect the dataset (shape, dtypes, missing values)
- Analyze churn distribution and relationships with:
  - Tenure, monthly charges, contract type
  - Internet service, payment method, tech support
- Correlation heatmap for numeric features
- Contract-level churn rates (month-to-month vs annual contracts)

### 2. Data cleaning

- Convert `Total Charges` to numeric; fill 11 missing values with `0` (new customers)
- Drop ID, location, and leaky/post-churn columns (`Churn Label`, `Churn Score`, `CLTV`, `Churn Reason`)
- **Drop `City` before encoding** (~1,100 categories removed — high cardinality, weak signal)

### 3. Feature engineering

Engineered features added before one-hot encoding:

| Feature | Definition |
|---------|------------|
| `Avg Monthly Spend` | `Total Charges / max(Tenure Months, 1)` |
| `Internet Add-ons` | Count of active streaming/security services |
| `Auto Payment` | 1 if bank/card automatic payment, else 0 |
| `Contract Ordinal` | Month-to-month = 0, one-year = 1, two-year = 2 |
| `Tenure Bucket` | New (0–12 mo), Mid (13–36), Loyal (37+) |

Then one-hot encode remaining categoricals with `pd.get_dummies(drop_first=True)` → **36 features** total.

### 4. Machine learning — churn prediction

- **Stratified 80/20 train/test split** (`random_state=42`)
- **Model comparison:** Logistic Regression vs Random Forest vs Gradient Boosting
- **Class imbalance:** `class_weight='balanced'`
- **Hyperparameter tuning:** `GridSearchCV` on **train only** (`scoring='recall'`, 5-fold stratified CV) — no test-set leakage
- **Best model:** `RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced')`
- **Threshold tuning:** probability cutoff ~**0.567** chosen for ~80% train recall (better for retention than default 0.5)
- Evaluation: accuracy, recall, precision, F1, ROC-AUC, confusion matrix, feature importance

### 5. Customer segmentation

- Build segments from tenure, monthly charges, total charges, and model-predicted churn probability
- Scale features with `StandardScaler`
- K-Means clustering (`k=3`, elbow method used to choose k)
- Segments labeled:
  - **Budget loyal customer**
  - **High risk new customer**
  - **Loyal premium customer**

### 6. Deployment

- Final cell in `Churn.ipynb` saves models to `models/`
- `app.py` loads artifacts and scores customer input from a form

## Model performance

Metrics from the **final Random Forest** on the **held-out 20% test set** (1,409 customers), after feature engineering and `GridSearchCV`.

| Metric | Value |
|--------|-------|
| **Accuracy** | **76.9%** |
| **Recall (churn class)** | **77.5%** |
| **Precision (churn class)** | 54.6% |
| **F1 (churn class)** | 64.1% |
| **ROC-AUC** | 0.853 |

**5-fold cross-validation** (train set, recall scoring):

| Metric | Value |
|--------|-------|
| CV recall | 78.6% |
| Best params | `n_estimators=100`, `max_depth=8` |

The model is tuned to **prioritize catching churners** (high recall) over overall accuracy. Precision is lower because false alarms are acceptable when the cost of missing a churner is higher.

> The deployed model in `app.py` is refit on **all** data before saving. Test metrics above reflect generalization on unseen data during development.

## Streamlit app

| Feature | Description |
|---------|-------------|
| **Predict tab** | Customer form → churn probability, risk band, segment, top drivers, retention action |
| **About model tab** | ML methodology, engineered features, global feature importance chart |
| **Sidebar** | Model metrics snapshot + quick profile presets |
| **Risk bands** | Low / Medium / High based on tuned threshold (~56.7%) |
| **Presets** | At-risk new customer, Loyal long-term, High-spend fiber |

### Preset smoke test

| Preset | Churn probability | Risk band |
|--------|-------------------|-----------|
| At-risk new customer | ~63% | High |
| High-spend fiber | ~52% | Medium |
| Loyal long-term | ~19% | Low |

## Setup

### Live app

**[Telco Churn Predictor](https://churn-predictor-rytj2prxn7fguqv888j6rc.streamlit.app/)**

> App loads `models/*.joblib` only — no Excel file needed at runtime. Retrain via `Churn.ipynb` if you change the model.

### Local setup

#### 1. Clone repository

```bash
git clone https://github.com/Rudranil-Datta/Churn-Predictor.git
cd Churn-Predictor
```

#### 2. Create virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

#### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Add dataset

Place `Telco_customer_churn.xlsx` in the project root (required for notebook training only).

#### 5. Train and export models — run notebook

```bash
jupyter notebook Churn.ipynb
```

Select your virtualenv kernel → run all cells through training → run the final **Save Models for Deployment** cell.

Expected output:

```text
Saved models with best params: {'max_depth': 8, 'n_estimators': 100}
Churn threshold: 0.567
```

This writes:

- `models/churn_model.joblib`
- `models/feature_columns.joblib`
- `models/churn_threshold.joblib`
- `models/feature_importance.joblib`
- `models/scaler.joblib`
- `models/kmeans.joblib`
- `models/cluster_names.joblib`

#### 6. Run Streamlit app locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

Fill in customer details (demographics, services, contract, billing) and click **Predict churn** to get churn probability, risk band, segment, drivers, and a recommended retention action.


## Tech stack

- Python 3.10, pandas, NumPy
- scikit-learn (Random Forest, GridSearchCV, K-Means, StandardScaler)
- matplotlib, seaborn (EDA in notebook)
- Streamlit (frontend)
- joblib (model persistence)
