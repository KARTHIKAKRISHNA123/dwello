# HomeVista Predictor
## AI-Powered Residential Property Valuation

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://housepricepredictiongit-8mdn7urrzjswzwhmgirivw.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)](./LICENSE)

A supervised machine learning application that predicts residential property sale prices using a Linear Regression pipeline. The project covers the complete ML lifecycle — from raw data ingestion and preprocessing through model training, evaluation, serialization, and cloud deployment via Streamlit Community Cloud.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Live Application](#2-live-application)
3. [Repository Structure](#3-repository-structure)
4. [Dataset](#4-dataset)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Model Architecture](#7-model-architecture)
8. [Training and Evaluation](#8-training-and-evaluation)
9. [Model Serialization](#9-model-serialization)
10. [Streamlit Application](#10-streamlit-application)
11. [Local Installation and Usage](#11-local-installation-and-usage)
12. [Cloud Deployment](#12-cloud-deployment)
13. [Known Issues and Resolutions](#13-known-issues-and-resolutions)
14. [Tech Stack](#14-tech-stack)
15. [References](#15-references)
16. [Author](#16-author)

---

## 1. Project Overview

**HomeVista Predictor** is an end-to-end supervised machine learning project that estimates the market sale price of a residential property from its physical, structural, and zoning attributes.

**Problem Statement**

Manual property appraisals are slow, inconsistent, and expensive. HomeVista Properties — the fictional real estate firm this project was designed for — required a data-driven system capable of producing reliable, instant valuations without human intervention.

**Objective**

Build a regression model that accepts structured property inputs and returns a predicted sale price in USD, served through an interactive web interface accessible to non-technical stakeholders.

**Scope**

- Machine learning task: Supervised regression (continuous target variable)
- Algorithm: Linear Regression via scikit-learn
- Interface: Streamlit web application
- Deployment target: Streamlit Community Cloud

**Methodology Summary**

The project follows the standard ML workflow: data loading, statistical inspection, missing value handling, feature engineering, pipeline construction, model training, performance evaluation, artifact export, and cloud deployment.

---

## 2. Live Application

**URL**: [https://housepriceprediction-uyw5rbwnmt97bnwbtcp8yc.streamlit.app/](https://housepriceprediction-uyw5rbwnmt97bnwbtcp8yc.streamlit.app/)

The live application accepts property specifications through a structured form and returns an estimated market valuation in real time. The interface is designed to be usable without any knowledge of the underlying machine learning model.

---

## 3. Repository Structure

```
HousePricePrediction/Mine/
│
├── app.py                        Primary Streamlit application — UI, input handling, inference
├── House_Price_Predictor.ipynb   Full training notebook — EDA, preprocessing, model, evaluation
├── HousePricePrediction.csv      Raw dataset (2,919 rows, 13 columns)
├── house_price_model.pkl         Serialized trained sklearn Pipeline (pickle format)
└── README.md                     Project documentation (this file)
```

**File Responsibilities**

| File | Role |
|------|------|
| `House_Price_Predictor.ipynb` | Research and training environment. Contains all data exploration, preprocessing logic, model fitting, and evaluation. Running all cells regenerates `house_price_model.pkl`. |
| `app.py` | Production inference script. Loads the saved pipeline, renders the UI, collects user inputs, constructs the input DataFrame, calls `model.predict()`, and displays the result. |
| `HousePricePrediction.csv` | Source dataset. Used exclusively in the notebook during training. Not loaded at runtime by the app. |
| `house_price_model.pkl` | The trained `sklearn.pipeline.Pipeline` object saved with `pickle`. This file must be present for the app to function and must be committed to the repository for Streamlit Cloud deployment. |

---

## 4. Dataset

**File**: `HousePricePrediction.csv`  
**Source**: Adapted from the Ames Housing Dataset (Kaggle).
**Total Rows**: 2,919  
**Rows with Target Label**: 1,460 (used for supervised training)  
**Target Variable**: `SalePrice` — continuous, denominated in USD

### Column Reference

| Column | Data Type | Category | Description |
|--------|-----------|----------|-------------|
| `Id` | Integer | Identifier | Unique row identifier. Dropped before training — no predictive value. |
| `MSSubClass` | Integer (treated as Categorical) | Structural | Building class code. 20 = 1-Story built 1946+, 30 = 1-Story older, 60 = 2-Story, 90 = Duplex, etc. |
| `MSZoning` | String | Zoning | General zoning classification: RL (Residential Low Density), RM (Medium), RH (High), C (Commercial), FV (Floating Village). |
| `LotArea` | Integer | Site | Lot size in square feet. |
| `LotConfig` | String | Site | Lot configuration: Inside, Corner, CulDSac, FR2 (two-sided frontage), FR3 (three-sided). |
| `BldgType` | String | Structural | Dwelling type: 1Fam (Single Family), 2fmCon (Two-Family Conversion), Duplex, TwnhsE (Townhouse End Unit), Twnhs (Inside Unit). |
| `OverallCond` | Integer | Condition | Overall condition rating on a 1–10 integer scale. 1 = Very Poor, 5 = Average, 10 = Very Excellent. |
| `YearBuilt` | Integer | Temporal | Original year of construction. |
| `YearRemodAdd` | Integer | Temporal | Year of most recent remodel. Equals `YearBuilt` if no remodel has occurred. |
| `Exterior1st` | String | Material | Primary exterior covering material: VinylSd (Vinyl Siding), MetalSd (Metal), BrkFace (Brick Face), CemntBd (Cement Board), Plywood, Stucco, and others. |
| `BsmtFinSF2` | Float | Area | Finished basement area of Type 2 finish in square feet. 0 indicates no Type 2 finish. |
| `TotalBsmtSF` | Float | Area | Total basement area in square feet. |
| `SalePrice` | Integer | **Target** | **Final sale price of the property in USD.** Present only in training rows. |

### Statistical Summary (Numerical Features)

| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| LotArea | 10,168 sq ft | 7,887 | 1,300 | 215,245 |
| OverallCond | 5.56 | 1.11 | 1 | 10 |
| YearBuilt | 1971 | 30.3 | 1872 | 2010 |
| YearRemodAdd | 1984 | 20.9 | 1950 | 2010 |
| BsmtFinSF2 | 49.6 sq ft | 169.2 | 0 | 1,526 |
| TotalBsmtSF | 1,051 sq ft | 440.8 | 0 | 6,110 |
| SalePrice | $180,921 | $79,443 | $34,900 | $755,000 |

> **Note on MSSubClass**: Although stored as an integer in the CSV file, `MSSubClass` encodes discrete categorical labels, not a continuous or ordinal quantity. It is cast to `str` before preprocessing and one-hot encoded accordingly.

---

## 5. Exploratory Data Analysis

The following analyses were performed in `House_Price_Predictor.ipynb` prior to model construction.

**Structural Inspection**

- Confirmed dataset shape (2,919 rows x 13 columns) using `df.shape`
- Examined data types with `df.dtypes` to identify type mismatches (e.g., `MSSubClass` as int)
- Reviewed descriptive statistics via `df.describe()` for all numeric columns

**Missing Value Assessment**

- `BsmtFinSF2` and `TotalBsmtSF` each have one missing row (2,918 records vs. 2,919)
- All other columns are complete
- Missing values are handled downstream via `SimpleImputer` inside the pipeline

**Target Variable Distribution**

- `SalePrice` ranges from $34,900 to $755,000 with a mean of $180,921
- Distribution is right-skewed, as is typical for property price data

**Categorical Cardinality**

| Feature | Unique Values |
|---------|--------------|
| MSSubClass | 15 |
| MSZoning | 5 |
| LotConfig | 5 |
| BldgType | 5 |
| Exterior1st | 15 |

High cardinality in `MSSubClass` and `Exterior1st` was handled through `OneHotEncoder` with `handle_unknown='ignore'` to suppress errors on unseen categories at inference time.

---

## 6. Data Preprocessing

All preprocessing is encapsulated within a single `sklearn.pipeline.Pipeline`, ensuring that the same transformations applied during training are automatically applied at inference time.

### Step 1 — Drop Identifier Column

`Id` is removed before any processing. It is a row counter with no relationship to sale price.

```python
df = df.drop(columns=['Id'])
```

### Step 2 — Cast MSSubClass to String

```python
df['MSSubClass'] = df['MSSubClass'].astype(str)
```

This ensures `MSSubClass` is recognized as a categorical variable rather than a numeric one during encoding.

### Step 3 — Define Feature Columns

```python
categorical_features = ['MSSubClass', 'MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']

numerical_features = ['LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                      'BsmtFinSF2', 'TotalBsmtSF']
```

### Step 4 — Build the ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', categorical_pipeline, categorical_features),
    ('num', numerical_pipeline, numerical_features)
])
```

### Step 5 — Assemble Final Pipeline

```python
from sklearn.linear_model import LinearRegression

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
```

### Preprocessing Summary

| Feature | Transformation Applied |
|---------|----------------------|
| MSSubClass | Cast to str, SimpleImputer (mode), OneHotEncoder |
| MSZoning | SimpleImputer (mode), OneHotEncoder |
| LotConfig | SimpleImputer (mode), OneHotEncoder |
| BldgType | SimpleImputer (mode), OneHotEncoder |
| Exterior1st | SimpleImputer (mode), OneHotEncoder |
| LotArea | SimpleImputer (mean), StandardScaler |
| OverallCond | SimpleImputer (mean), StandardScaler |
| YearBuilt | SimpleImputer (mean), StandardScaler |
| YearRemodAdd | SimpleImputer (mean), StandardScaler |
| BsmtFinSF2 | SimpleImputer (mean), StandardScaler |
| TotalBsmtSF | SimpleImputer (mean), StandardScaler |

---

## 7. Model Architecture

**Algorithm**: `sklearn.linear_model.LinearRegression`

Linear Regression models the relationship between features and the target as a linear combination:

```
SalePrice = w1*x1 + w2*x2 + ... + wn*xn + bias
```

Coefficients are estimated by minimizing the Ordinary Least Squares (OLS) objective — the sum of squared residuals between predicted and actual prices.

**Why Linear Regression**

- Interpretable and auditable — coefficients show the direction and magnitude of each feature's effect on price
- No hyperparameter tuning required, making the baseline robust and reproducible
- Computationally efficient on a dataset of this size
- Appropriate as the initial benchmark before exploring regularized variants (Ridge, Lasso)

**Full Pipeline Flow**

```
Raw Input DataFrame (11 features)
           |
           v
   ColumnTransformer
   |                |
   v                v
Categorical       Numerical
Pipeline          Pipeline
   |                |
   v                v
SimpleImputer    SimpleImputer
(mode)           (mean)
   |                |
   v                v
OneHotEncoder    StandardScaler
   |                |
   +-------+--------+
           |
           v
   Encoded and Scaled Feature Matrix
           |
           v
   LinearRegression.fit()
           |
           v
   Trained Coefficient Vector
           |
           v
   Predicted SalePrice (USD)
```

---

## 8. Training and Evaluation

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Training set: 1,168 samples (80%)
- Test set: 289 samples (20%)
- `random_state=42` ensures reproducibility across runs

### Model Fitting

```python
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
```

### Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R-squared (R2) | 0.6673 | The model accounts for approximately 66.7% of variance in SalePrice |
| Mean Absolute Error (MAE) | $30,593 | On average, predictions deviate from actual prices by approximately $30,000 |
| Root Mean Squared Error (RMSE) | $45,389 | Penalizes large errors more than MAE; typical prediction error magnitude is approximately $45,000 |

### Residual Analysis

| Statistic | Value |
|-----------|-------|
| Residual Mean | -$519 (near-zero — low systematic bias) |
| Residual Std Dev | $45,465 |
| Residual Min | -$332,713 |
| Residual Max | +$105,684 |

The distribution of residuals is approximately bell-shaped and centered near zero, confirming that the model does not systematically over- or under-predict across the test set.

The model performs well on mid-range properties (roughly $100,000 to $300,000), which constitute the majority of training data. Accuracy may degrade on extreme-value properties outside this range due to the limited expressive capacity of linear models.

---

## 9. Model Serialization

The trained pipeline is serialized using Python's `pickle` module:

```python
import pickle

with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)
```

At inference time in `app.py`, the model is deserialized:

```python
@st.cache_resource
def load_model():
    with open('house_price_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
```

`@st.cache_resource` ensures the model is loaded from disk only once per session, preventing repeated deserialization on each user interaction and improving application responsiveness.

**What the .pkl file contains**

The `.pkl` file contains the complete `Pipeline` object — including all fitted `ColumnTransformer` parameters (encoder vocabularies, scaler means, and scaler variances) and all LinearRegression coefficients. The column schema baked into this artifact at training time must exactly match the column schema provided at inference time.

---

## 10. Streamlit Application

### Application Architecture

`app.py` is structured in six logical sections:

1. **Page configuration** — title, layout, sidebar state
2. **Custom CSS injection** — Vercel-inspired dark theme with Inter font, smooth input animations, and button hover transitions
3. **Model loading** — cached deserialization of `house_price_model.pkl`
4. **UI form construction** — two-column layout with dropdowns, number inputs, and a slider
5. **Inference logic** — input dictionary construction, DataFrame creation, `model.predict()` call
6. **Result display** — formatted dollar output with loading spinner and toast notification

### Input Fields

| Field | Widget Type | Valid Range or Options |
|-------|------------|----------------------|
| Building Class | Selectbox | 15 options (e.g., 1-Story 1946+, 2-Story, Duplex) |
| Zoning Classification | Selectbox | RL, RM, C (all), FV, RH |
| Dwelling Type | Selectbox | Single-Family, Two-Family, Duplex, Townhouse End, Townhouse Inside |
| Lot Configuration | Selectbox | Inside, Corner, Cul-de-sac, Frontage 2 Sides, Frontage 3 Sides |
| Exterior Material | Selectbox | 15 options (Vinyl, Metal, Wood, Brick, Cement, etc.) |
| Lot Area (sq ft) | Number Input | 1,000 to 200,000 |
| Total Basement (sq ft) | Number Input | 0 to 6,000 |
| Type 2 Finished Bsmt (sq ft) | Number Input | 0 to 1,500 |
| Overall Condition | Slider | 1 (Very Poor) to 10 (Very Excellent) |
| Construction Year | Number Input | 1800 to 2024 |
| Remodel Year | Number Input | 1800 to 2024 |

### Inference Flow

```python
input_data = {
    'MSSubClass':   mssubclass,
    'MSZoning':     mszoning,
    'LotArea':      lotarea,
    'LotConfig':    lotconfig,
    'BldgType':     bldgtype,
    'OverallCond':  overallcond,
    'YearBuilt':    yearbuilt,
    'YearRemodAdd': yearremodadd,
    'Exterior1st':  exterior1st,
    'BsmtFinSF2':   bsmtfinsf2,
    'TotalBsmtSF':  totalbsmtsf
}

input_df = pd.DataFrame([input_data])
predicted_price = model.predict(input_df)[0]
```

The column names in `input_data` must exactly match those used during model training. Any mismatch in name, type, or presence will cause the `ColumnTransformer` to raise a `ValueError`.

### Output

Upon clicking **Generate Valuation**, the application displays:

```
Estimated Market Value: $XXX,XXX.XX
```

---

## 11. Local Installation and Usage

### System Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/KARTHIKAKRISHNA123/House_Price_Prediction.git
cd "House_Price_Prediction/Mine"
```

### Step 2 — Install Dependencies

```bash
pip install streamlit pandas scikit-learn numpy
```

For notebook usage (training and EDA):

```bash
pip install jupyter matplotlib seaborn
```

### Step 3 — Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501` in your default browser.

### Step 4 — Retrain the Model (Optional)

If you modify the dataset or preprocessing logic, regenerate the model artifact:

```bash
jupyter notebook House_Price_Predictor.ipynb
```

Run all cells in sequence. The final cell will overwrite `house_price_model.pkl` with the newly trained pipeline. Restart the Streamlit application to load the updated model.

### Dependency Reference

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| streamlit | 1.20 | Web application framework |
| pandas | 1.5 | Data manipulation and DataFrame construction |
| scikit-learn | 1.0 | ML pipeline, preprocessing, Linear Regression |
| numpy | 1.23 | Numerical operations |
| matplotlib | 3.6 | Plotting (notebook only) |
| seaborn | 0.12 | Statistical visualization (notebook only) |

---

## 12. Cloud Deployment

**Platform**: Streamlit Community Cloud  
**Live URL**: [housepricepredictiongit-8mdn7urrzjswzwhmgirivw.streamlit.app](https://housepricepredictiongit-8mdn7urrzjswzwhmgirivw.streamlit.app)

### Deployment Prerequisites

A public GitHub repository containing:

- `app.py` — the Streamlit entry point
- `house_price_model.pkl` — the serialized model artifact
- `requirements.txt` — recommended for reproducible builds

### Recommended requirements.txt

```
streamlit
pandas
scikit-learn
numpy
```

### Deployment Steps

1. Push the repository to GitHub. Ensure `house_price_model.pkl` is committed and not listed in `.gitignore`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. Click **New app**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Click **Deploy**.

Streamlit Community Cloud will clone the repository, install dependencies, execute `streamlit run app.py`, and serve the application at a public HTTPS URL.

### Operational Notes

- The `house_price_model.pkl` file must be committed to the repository. Streamlit Cloud has no mechanism to run the training notebook at deploy time — it only serves the app. If the `.pkl` file is missing or corrupt, the application will fail on startup.
- Any changes pushed to the connected branch automatically trigger a redeploy.
- Free tier applications may sleep after a period of inactivity and require a brief cold-start on the next visit.

---

## 13. Known Issues and Resolutions

### ValueError — columns are missing: {'HouseAge'}

**Symptom**

```
ValueError: columns are missing: {'HouseAge'}
Traceback:
  File "app.py", line 157, in <module>
    predicted_price = model.predict(input_df)[0]
  File "sklearn/pipeline.py", in predict
    Xt = transform.transform(Xt)
  File "sklearn/compose/_column_transformer.py", in transform
    raise ValueError(f"columns are missing: {diff}")
```

**Root Cause**

The `ColumnTransformer` inside the trained pipeline remembers the exact column names it was fitted on during training. If the notebook computed a derived feature — for example, `df['HouseAge'] = 2024 - df['YearBuilt']` — and the model was trained on `HouseAge`, but `app.py` passes the raw `YearBuilt` column instead, the column schemas do not match and the pipeline raises this error.

This is a schema consistency requirement, not a bug. The feature set at inference time must be identical to the feature set at training time, including column names and any derived features.

**Resolution — Option A (Preferred): Align app.py to the trained model**

Add the derived feature to the input dictionary in `app.py` using the exact same formula used in the notebook:

```python
input_data = {
    ...
    'HouseAge': 2024 - yearbuilt,
    ...
}
```

Remove `YearBuilt` and/or `YearRemodAdd` from `input_data` if they were not included as standalone features during training.

**Resolution — Option B: Retrain without derived features**

If using `YearBuilt` and `YearRemodAdd` as raw inputs in the app is preferred, remove the `HouseAge` computation from the notebook, retrain the pipeline, and replace `house_price_model.pkl`.

**Prevention**

Define the list of feature column names as a shared constant. Keep the training notebook and inference script in strict agreement on the column schema. Never rely on implicit alignment between two independently maintained files.

---

## 14. Tech Stack

| Technology | Role |
|------------|------|
| Python 3.x | Core programming language |
| pandas | Data loading, manipulation, DataFrame construction |
| NumPy | Numerical operations and array handling |
| scikit-learn | ColumnTransformer, Pipeline, OneHotEncoder, StandardScaler, SimpleImputer, LinearRegression |
| Streamlit | Web application UI and deployment framework |
| pickle | Model serialization and deserialization |
| Matplotlib | EDA plots and residual visualization (notebook only) |
| Seaborn | Distribution plots for residual analysis (notebook only) |
| Jupyter Notebook | Training and experimentation environment |
| Git | Version control |
| Streamlit Community Cloud | Cloud hosting and application deployment |

---

## 15. References

- **Project Context**: Supervised Machine Learning — Linear Regression, AI/ML.
- **Dataset Origin**: Adapted from the Ames Housing Dataset, originally compiled by Dean De Cock (2011) and hosted on Kaggle
- **scikit-learn Pipeline**: [https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- **scikit-learn ColumnTransformer**: [https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- **Streamlit Deployment Guide**: [https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

## 16. Author

**Karthika Krishna**  
CSE Student | Full-Stack Developer | Passionate in AI/ML

[![GitHub](https://img.shields.io/badge/GitHub-KARTHIKAKRISHNA123-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/KARTHIKAKRISHNA123)

---

*This project is part of the [aiml](https://github.com/KARTHIKAKRISHNA123/aiml) parent workspace and is maintained as an independent module within that repository.*