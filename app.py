import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# =========================================================
# 1. Load Artifacts
# =========================================================
ARTIFACT_DIR = Path("models") / "demo"

model = joblib.load(ARTIFACT_DIR / "demo_model.pkl")

with open(ARTIFACT_DIR / "demo_threshold.json", "r", encoding="utf-8") as f:
    threshold = json.load(f)["threshold"]

with open(ARTIFACT_DIR / "demo_feature_columns.json", "r", encoding="utf-8") as f:
    feature_columns = json.load(f)["feature_columns"]

# =========================================================
# 2. Page Config
# =========================================================
st.set_page_config(page_title="Loan Approval Prediction Demo", layout="centered")

st.title("Loan Approval Prediction Demo")
st.write(
    """
This application estimates the probability that a loan application will be approved
based only on the information entered below.

This is a decision-support prototype for demonstration purposes.
It is designed to support judgment, not replace human decision-making.
"""
)

st.markdown("---")

# =========================================================
# 3. Input Section
# =========================================================
st.subheader("Applicant Information")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=680,
    step=1
)

annual_income = st.number_input(
    "Annual Income ($)",
    min_value=0.0,
    value=60000.0,
    step=1000.0
)

loan_amount = st.number_input(
    "Loan Amount ($)",
    min_value=0.0,
    value=20000.0,
    step=1000.0
)

loan_duration = st.number_input(
    "Loan Duration (months)",
    min_value=1,
    max_value=360,
    value=36,
    step=1
)

monthly_debt_payments = st.number_input(
    "Monthly Debt Payments ($)",
    min_value=0.0,
    value=500.0,
    step=50.0
)

savings_balance = st.number_input(
    "Savings Account Balance ($)",
    min_value=0.0,
    value=5000.0,
    step=500.0
)

checking_balance = st.number_input(
    "Checking Account Balance ($)",
    min_value=0.0,
    value=2000.0,
    step=500.0
)

employment_status = st.selectbox(
    "Employment Status",
    options=["Employed", "Self-Employed", "Unemployed"]
)

previous_loan_defaults = st.selectbox(
    "Previous Loan Defaults",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

bankruptcy_history = st.selectbox(
    "Bankruptcy History",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

st.markdown("---")

# =========================================================
# 4. Build Input Data
# =========================================================
input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "AnnualIncome": annual_income,
    "LoanAmount": loan_amount,
    "LoanDuration": loan_duration,
    "MonthlyDebtPayments": monthly_debt_payments,
    "SavingsAccountBalance": savings_balance,
    "CheckingAccountBalance": checking_balance,
    "EmploymentStatus": employment_status,
    "PreviousLoanDefaults": previous_loan_defaults,
    "BankruptcyHistory": bankruptcy_history,
}])

input_data = input_data[feature_columns]

# =========================================================
# 5. Prediction
# =========================================================
if st.button("Run Prediction"):
    probability = model.predict_proba(input_data)[0, 1]
    decision = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")

    # probability display
    display_prob = probability * 100

    if display_prob < 1:
        display_text = "< 1%"
    elif display_prob > 99:
        display_text = "> 99%"
    else:
        display_text = f"{display_prob:.1f}%"

    st.metric("Approval Probability", display_text)

    st.write(f"**Decision Threshold:** {threshold:.2f}")

    if decision == 1:
        st.success("Final Decision: Approve")
    else:
        st.error("Final Decision: Reject")

    st.markdown("### Interpretation")
    st.write(
        """
This probability shows how likely the application is to be approved based on the information entered above.

The final decision is made by comparing the predicted probability to the decision threshold.
If the probability is higher than the threshold, the application is classified as approve.
If it is lower, the application is classified as reject.

This result should be used as a support tool, not as a guaranteed final answer.
"""
    )

    st.markdown("### Input Summary")
    st.dataframe(input_data, use_container_width=True)

    st.markdown("### Important Limitations")
    st.write(
        """
- This demo uses a reduced number of features to keep the app simple and easy to use.
- It does not include every factor that could affect a real loan decision.
- The prediction is based on patterns in the training data, so it is not a guarantee.
- This tool is meant to support decision-making, not replace it completely.
"""
    )