# Insurance Premium Prediction using Polynomial Regression (Version 2.0)

## 📖 Project Overview

This project builds an end-to-end Machine Learning pipeline to predict insurance premiums using Polynomial Regression. 

The project includes:

- Data preprocessing
- IQR-based outlier capping
- Exploratory Data Analysis (EDA)
- Polynomial Regression with Pipeline
- K-Fold Cross Validation
- Model Evaluation using multiple metrics
- Residual diagnostics
- Model persistence

This upgraded version focuses on proper ML workflow and engineering practices.

---

## 📂 Dataset

The dataset contains insurance-related features such as:

- Age
- BMI
- Number of Children
- Smoker Status
- Income
- Exercise Level
- Region Risk
- Pre-existing Conditions
- Premium (Target Variable)

---

## 🔍 Exploratory Data Analysis (EDA)

The following steps were performed:

- Head & Tail inspection
- Shape and column analysis
- Data types check
- Missing value check
- Duplicate check
- Distribution plots (Histograms + KDE)
- Boxplots for outlier visualization
- Correlation heatmap

---

## 🛠 Outlier Handling

Instead of removing outliers, IQR-based **capping** was applied:

Lower Bound = Q1 − 1.5 × IQR  
Upper Bound = Q3 + 1.5 × IQR  

This preserves dataset size while reducing extreme value influence.

---

## 🤖 Model Used

Polynomial Regression implemented using Scikit-learn Pipeline:

- StandardScaler
- PolynomialFeatures (degree = 2)
- LinearRegression

---

## 🔁 Cross Validation

5-Fold Cross Validation was applied using:

KFold (shuffle=True, random_state=42)

Evaluation metrics computed across folds:
- R² Score
- RMSE

---

## 📊 Evaluation Metrics

On Test Set:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 📉 Residual Analysis

A residual scatter plot was generated to:

- Check model bias
- Detect heteroscedasticity
- Evaluate prediction stability

---

## 📈 Polynomial Degree Comparison

Model performance was compared for:

- Degree 1 (Linear Regression)
- Degree 2
- Degree 3

This helps in selecting the optimal polynomial complexity.

---

## 💾 Model Saving

The trained model is saved using joblib:

insurance_polynomial_model.pkl

---

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/devendrakushwah80/insurance-premium-predictor-polynomial-regression.git
```
Install dependencies:

pip install -r requirements.txt

Run the notebook or script.

📦 Requirements

numpy

pandas

scikit-learn

matplotlib

seaborn

joblib

🎯 Key Learning Outcomes

Proper ML workflow implementation

Outlier capping using IQR

Polynomial feature engineering

Cross-validation best practices

Regression performance evaluation

Residual diagnostics

👨‍💻 Author

Devendra Kushwah
