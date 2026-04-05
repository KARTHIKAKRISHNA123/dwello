# 🏡 House Price Prediction
> A supervised machine learning project that predicts residential property sale prices using Linear Regression.

---

## 📌 Project Context

This project is built as part of my **AI/ML learning journe* fulfilling **Supervised ML Assignment 1** for *HomeVista Properties* — a fictional real estate firm looking to automate their house pricing process.

> **Role**: Machine Learning Engineer  
> **Goal**: Build a Linear Regression model that predicts the market `SalePrice` of a house based on its features.

---

## 📁 Repository Structure

```
House_Price_Prediction/
│
├── HousePricePrediction.csv       ← Dataset (13 columns, provided by course)
├── house_price_predictor.ipynb    ← My implementation notebook
├── lasso_regression.ipynb         ← Lasso regression exploration
├── model.pkl                      ← Saved trained model (after training)
├── requirements.txt               ← Python dependencies
└── README.md                      ← You are here
```

---

## 📊 Dataset Overview

**File**: `HousePricePrediction.csv`  
**Target Variable**: `SalePrice` (Continuous — market price of house in USD)

| Column | Type | Description |
|--------|------|-------------|
| `Id` | Identifier | Unique row ID — dropped before training |
| `MSSubClass` | Categorical* | Building class (20=1-Story, 60=2-Story, etc.) |
| `MSZoning` | Categorical | Residential zone type |
| `LotArea` | Numerical | Lot size in square feet |
| `LotConfig` | Categorical | Lot configuration (Inside, Corner, CulDeSac) |
| `BldgType` | Categorical | Type of dwelling (1Fam, Duplex, TwnhsE) |
| `OverallCond` | Numerical | Overall condition rating (1–10) |
| `YearBuilt` | Numerical | Original construction year |
| `YearRemodAdd` | Numerical | Remodel year |
| `Exterior1st` | Categorical | Exterior covering material |
| `BsmtFinSF2` | Numerical | Finished basement area (sq ft) |
| `TotalBsmtSF` | Numerical | Total basement area (sq ft) |
| `SalePrice` | **Target** | **House sale price in USD** |

> ⚠️ `MSSubClass` looks numerical but represents categories — treated as categorical.

---

## 🧪 Methodology

### 1. Data Preprocessing
- Dropped `Id` column (no predictive value)
- Cast `MSSubClass` from `int` → `str` (categorical treatment)
- **One-Hot Encoding** for: `MSZoning`, `LotConfig`, `BldgType`, `Exterior1st`, `MSSubClass`
- **StandardScaler** applied to numerical features to normalize scale differences

### 2. Model
- **Algorithm**: Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Pipeline**: `ColumnTransformer` → `LinearRegression` (end-to-end sklearn Pipeline)
- **Train/Test Split**: 80% training / 20% testing (`random_state=42`)

### 3. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **R² Score** | How much variance in SalePrice is explained by the model |
| **MAE** | Average dollar amount predictions are off by |
| **RMSE** | Penalizes larger errors more heavily than MAE |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas scikit-learn numpy jupyter
```

### Steps
```bash
# 1. Clone the repo
git clone https://github.com/KARTHIKAKRISHNA123/House_Price_Prediction.git
cd House_Price_Prediction

# 2. Launch Jupyter
jupyter notebook

# 3. Open house_price_predictor.ipynb and run all cells
```

---

## 🔮 Future Plans

- [ ] Build my own version from scratch inside the `Mine/` folder
- [ ] Add EDA (Exploratory Data Analysis) with visualizations
- [ ] Compare Linear Regression vs Lasso vs Ridge
- [ ] Export trained model as `model.pkl`
- [ ] Deploy using **Streamlit** on Streamlit Community Cloud
- [ ] Add feature importance chart

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?logo=jupyter)

---

## 📚 References

- Course Assignment: Supervised ML — Linear Regression
- Dataset: Provided by course (HomeVista Properties scenario)

---

## 👩‍💻 Author

**Karthika Krishna**  
CSE Student | Aspiring Full Stack Developer Passionate in AI/ML
[![GitHub](https://img.shields.io/badge/GitHub-KARTHIKAKRISHNA123-black?logo=github)](https://github.com/KARTHIKAKRISHNA123)

---

> 📌 *This project is part of the [aiml](https://github.com/KARTHIKAKRISHNA123/aiml) parent workspace — tracked as an independent Git submodule.*