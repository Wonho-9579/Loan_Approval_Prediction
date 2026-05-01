# Loan Approval Prediction using Machine Learning

## 1. Business Problem / Motivation

Financial institutions must evaluate loan applications efficiently while managing credit risk. Traditional manual review processes can be slow, inconsistent, and difficult to scale.

This project develops a machine learning-based decision support system that predicts whether a loan application is likely to be approved. The goal is to improve consistency, efficiency, and interpretability in the decision-making process.

---

## 2. Project Overview

This project builds an end-to-end machine learning pipeline for loan approval prediction.

Key components include:
- Data preprocessing and feature engineering
- Model training and comparison
- Threshold optimization
- Model evaluation
- Model interpretability
- Streamlit-based interactive demo

Final Result:
- Selected Model: Logistic Regression
- Test ROC-AUC: 0.9738
- Accuracy: 0.9143
- F1-score: 0.8334

---

## 3. Data

- Source:  
https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval

- Type: Tabular dataset  
- Size: 20,000 rows × 36 columns  

### Target Variable
- LoanApproved (0 = Reject, 1 = Approve)

### Key Features
- CreditScore
- AnnualIncome
- LoanAmount
- LoanDuration
- MonthlyDebtPayments
- SavingsAccountBalance
- CheckingAccountBalance
- EmploymentStatus
- PreviousLoanDefaults
- BankruptcyHistory

---

## 4. Data Preprocessing

Key preprocessing steps:
- Removed data leakage features:
  - ApplicationDate
  - InterestRate-related variables
  - RiskScore
- Train / Validation / Test split (70 / 15 / 15)
- Feature scaling for numeric variables
- One-hot encoding for categorical variables

---

## 5. Exploratory Data Analysis (EDA)

Key observations:
- Class imbalance exists (more rejections than approvals)
- Financial stability indicators strongly influence approval
- Debt-related variables are highly predictive

---

## 6. Modeling Approach

Three models were trained and compared:

- Logistic Regression (Baseline)
- Random Forest
- XGBoost (Advanced Model)

### Model Selection

Logistic Regression was selected as the final model because:
- Highest validation ROC-AUC (0.9776)
- Strong test performance
- High interpretability compared to complex models

---

## 7. Model Training

Tools used:
- Python (scikit-learn, xgboost)
- Pipeline for preprocessing + modeling

Cross-validation:
- 5-fold CV used for robustness

---

## 8. Results

### Validation Performance

| Model | ROC-AUC | Precision | Recall | F1 |
|------|--------|----------|--------|----|
| Logistic Regression | 0.9776 | 0.7573 | 0.9358 | 0.8372 |
| XGBoost | 0.9763 | 0.7793 | 0.9010 | 0.8357 |
| Random Forest | 0.9657 | 0.7605 | 0.8856 | 0.8183 |

---

### Final Test Performance

- ROC-AUC: 0.9738
- Accuracy: 0.9143
- Precision: 0.7785
- Recall: 0.8968
- F1-score: 0.8334

---

## 9. Model Interpretation

Model interpretability was achieved using Logistic Regression coefficients.

Key influential features include:
- MonthlyIncome
- LoanToIncomeRatio
- LoanAmount
- EmploymentStatus
- MonthlyDebtBurden

These features strongly influence the model's predictions.

---

## 10. Key Insights

- Income and debt ratio are the strongest predictors of loan approval
- Employment stability significantly impacts outcomes
- Logistic Regression performs competitively with more complex models
- Threshold tuning improves the balance between precision and recall

---

## 11. Conclusion

This project demonstrates that a simple and interpretable model can achieve high performance in loan approval prediction.

The pipeline is robust, reproducible, and suitable for real-world decision support.

---

## 12. Future Work

- Try additional advanced models (LightGBM, Neural Networks)
- Improve feature engineering
- Address class imbalance more aggressively
- Enhance model explainability
- Deploy as a full web application

---

## 13. Streamlit Demo Application

This project includes an interactive Streamlit app for real-time predictions.

Important:
- The full training pipeline uses all available features
- The Streamlit demo uses a reduced feature set for usability
- The demo model is designed for interaction, not full production use

---

## 14. How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run the Streamlit app

streamlit run app.py

---

## 15. Repository Structure

loan-approval-prediction/
├── app.py
├── README.md
├── requirements.txt
├── models/
├── results/
├── images/

---

## 16. Requirements

pip install -r requirements.txt
