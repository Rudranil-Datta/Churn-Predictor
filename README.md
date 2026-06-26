# Telco Customer Churn Predictor

Machine learning project that analyzes telecom customer data, predicts churn risk, and segments customers for retention targeting. Includes a Jupyter notebook for exploration and training, plus a Streamlit app for live predictions.

## Purpose

Help a telecom provider understand **why customers leave** and **which customers are most likely to churn**, so retention teams can act early. The workflow covers:

1. Exploratory analysis of churn drivers
2. Supervised churn classification with Random Forest
3. Unsupervised customer segmentation with K-Means
4. A Streamlit frontend for scoring new customer profiles

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
- One-hot encode categoricals with `pd.get_dummies(drop_first=True)`
- Split features (`X`) and target (`y` = `Churn Value`)

### 3. Machine learning — churn prediction

- **80/20 train/test split** (`random_state=42`)
- **Baseline:** `RandomForestClassifier(n_estimators=100)`
- **Class imbalance:** `class_weight='balanced'`
- **Tuned final model:** `n_estimators=300`, `max_depth=10`, `class_weight='balanced'`
- Evaluation: accuracy, confusion matrix, classification report, ROC-AUC
- Optional grid search over tree count and depth; 5-fold cross-validation on the final model
- Feature importance analysis; low-value features explored

### 4. Customer segmentation

- Build segments from tenure, monthly charges, total charges, and model-predicted churn probability
- Scale features with `StandardScaler`
- K-Means clustering (`k=3`, elbow method used to choose k)
- Segments labeled:
  - **Budget loyal customer**
  - **High risk new customer**
  - **Loyal premium customer**

### 5. Deployment

- Final cell in `Churn.ipynb` saves models to `models/`
- `app.py` loads artifacts and scores customer input from a form

## Model performance

Metrics below are from the **tuned Random Forest** (`rf_tuned`) on the **held-out 20% test set** (1,409 customers).

| Metric | Value |
|--------|-------|
| **Accuracy** | **74.2%** |
| **Recall (churn class)** | **81.2%** |
| **Precision (churn class)** | 52.9% |
| **F1 (churn class)** | 64.1% |
| **ROC-AUC** | 0.84 |

**5-fold cross-validation** (train set only):

| Metric | Mean |
|--------|------|
| Accuracy | 75.0% |
| Recall | 81.8% |

The tuned model trades some overall accuracy vs the ~80% baseline in exchange for **much higher churn recall** (81% vs ~53% baseline), which is better for catching customers who might leave.

> The deployed model in `app.py` is refit on **all** data before saving, so live predictions use the full dataset; test metrics above reflect generalization on unseen data during development.

## Setup

### Live app

**[Telco Churn Predictor](https://churn-predictor-imrepfchefx9j2cksr866g.streamlit.app/)** 

### Local setup

#### 1. Clone repository

```bash
git clone https://github.com/Rudranil-Datta/Churn-Predictor.git
cd churn-predictor
```

#### 2. Create virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
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
Saved to models/: churn_model, feature_columns, cities, scaler, kmeans, cluster_names
```

This writes:

- `models/churn_model.joblib`
- `models/feature_columns.joblib`
- `models/cities.joblib`
- `models/scaler.joblib`
- `models/kmeans.joblib`
- `models/cluster_names.joblib`

#### 6. Run Streamlit app locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

Fill in customer details (city, demographics, services, contract, billing) and click **Predict churn** to get churn probability, a churn/stay label, and a customer segment.

#### Smoke test

| Input | Expected |
|-------|----------|
| Month-to-month contract, electronic check, short tenure, fiber internet | Higher churn probability |
| Two-year contract, bank transfer, long tenure | Lower churn probability |
| Any valid profile | Segment label (budget loyal / high risk new / loyal premium) |


## Tech stack

- Python, pandas, NumPy
- scikit-learn (Random Forest, K-Means, StandardScaler)
- matplotlib, seaborn (EDA in notebook)
- Streamlit (frontend)
- joblib (model persistence)
