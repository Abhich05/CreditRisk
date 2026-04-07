# 🏦 Credit Risk Prediction System

An **industry-ready** machine learning project for predicting loan defaults with explainable AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)

## 🎯 Problem Statement

> Predict the probability that a loan applicant will default, using historical financial and behavioral data, and explain the decision for regulatory compliance.

## ✨ Features

- **3 ML Models**: Logistic Regression, Random Forest, XGBoost (primary)
- **Explainable AI**: SHAP values for global and local explanations
- **Interactive Dashboard**: Streamlit app with real-time predictions
- **Class Imbalance Handling**: Weighted models for skewed data
- **Feature Engineering**: Industry-relevant risk indicators

## 📁 Project Structure

```
credit_risk/
├── data/
│   ├── raw/              # Original loan_data.csv
│   └── processed/        # Cleaned train/val/test splits
├── models/               # Saved model artifacts (.pkl)
├── notebooks/            # EDA and experiments
├── src/
│   ├── __init__.py
│   ├── preprocessing.py  # Data cleaning & feature engineering
│   ├── model.py          # Model training & evaluation
│   └── explain.py        # SHAP explanations
├── app.py                # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Run Preprocessing

```bash
python -m src.preprocessing
```

### 3. Train Models

```bash
python -m src.model
```

### 4. Generate Explanations

```bash
python -m src.explain
```

### 5. Launch Dashboard

```bash
streamlit run app.py
```

## 📊 Dataset

- **Source**: Lending Club Loan Data
- **Rows**: ~9,600 samples
- **Target**: `not_fully_paid` (0=Paid, 1=Default)
- **Features**: 14 original + 5 engineered

### Key Features

| Feature | Description |
|---------|-------------|
| `fico` | Credit score (300-850) |
| `dti` | Debt-to-income ratio |
| `int_rate` | Interest rate |
| `revol_util` | Credit utilization % |
| `delinq_2yrs` | Past delinquencies |

## 📈 Model Performance

| Model | ROC-AUC | Precision | Recall |
|-------|---------|-----------|--------|
| Logistic Regression | ~0.67 | ~0.25 | ~0.60 |
| Random Forest | ~0.70 | ~0.28 | ~0.55 |
| **XGBoost** | **~0.72** | **~0.30** | **~0.58** |

## 🔍 Explainability

### Global Feature Importance
- FICO score is the strongest predictor
- DTI and interest rate follow closely
- Delinquency history impacts risk significantly

### Local Explanations
Each prediction includes:
- Default probability percentage
- Risk category (Low/Medium/High)
- Top 5 contributing factors with SHAP values

## 💡 Interview Talking Points

1. **Data Leakage Prevention**: Removed post-loan features
2. **Class Imbalance**: Used `scale_pos_weight` in XGBoost
3. **Explainability**: SHAP for regulatory compliance (Basel III/GDPR)
4. **Feature Engineering**: Business-driven risk indicators
5. **Model Comparison**: Baseline (LR) vs. ensemble (RF, XGBoost)

## 🛠️ Technologies

- **Python 3.10+**
- **pandas, numpy** - Data manipulation
- **scikit-learn** - Preprocessing, LR, RF
- **XGBoost** - Primary model
- **SHAP** - Explainability
- **Streamlit** - Dashboard
- **Plotly** - Visualizations

## 📝 License

This project is for educational purposes.

---

**Built for Credit Risk ML Interviews** 🎯
