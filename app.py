# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from PIL import Image
import io

# Load models and scaler
pd_model = joblib.load('pd_model.joblib')
lgd_model = joblib.load('lgd_model.joblib')
ead_model = joblib.load('ead_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define features
pd_features = ['loan_amnt', 'int_rate', 'dti', 'annual_inc', 'revol_bal', 'revol_util', 
               'delinq_2yrs', 'open_acc', 'total_acc', 'grade', 'term', 'home_ownership', 
               'purpose', 'verification_status']
lgd_ead_features = ['loan_amnt', 'total_pymnt', 'recoveries', 'out_prncp', 'int_rate', 'dti', 
                    'grade', 'term', 'home_ownership']

# Streamlit app
st.title("Credit Risk Management Dashboard")
st.write("Analyze loan risk using PD, LGD, EAD, and EL. Input loan details or upload a CSV for batch predictions.")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Single Loan Prediction", "Batch Prediction", "Visualizations", "Portfolio Analysis"])

if page == "Home":
    st.header("Welcome to the Credit Risk Dashboard")
    st.write("""
    This app predicts credit risk for LendingClub loans using:
    - **Probability of Default (PD)**: Logistic regression model.
    - **Loss Given Default (LGD)** and **Exposure at Default (EAD)**: Linear regression models.
    - **Expected Loss (EL)**: Calculated as EL = PD × LGD × EAD.
    - **WoE/IV**: For feature selection and transformation.
    - **SHAP**: For AI interpretability (XGBoost).
    
    Navigate to:
    - **Single Loan Prediction**: Enter loan details for individual risk assessment.
    - **Batch Prediction**: Upload a CSV for multiple loans.
    - **Visualizations**: View model performance and feature importance.
    - **Portfolio Analysis**: Summarize portfolio risk.
    """)

elif page == "Single Loan Prediction":
    st.header("Single Loan Prediction")
    st.write("Enter loan details to predict PD, LGD, EAD, and EL.")

    # Input form
    with st.form("loan_form"):
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000)
        int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=10.0) / 100
        dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=20.0)
        annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=60000)
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=100000, value=5000)
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=30.0) / 100
        delinq_2yrs = st.number_input("Delinquencies (Past 2 Years)", min_value=0, max_value=10, value=0)
        open_acc = st.number_input("Open Accounts", min_value=0, max_value=50, value=10)
        total_acc = st.number_input("Total Accounts", min_value=0, max_value=100, value=20)
        grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        term = st.selectbox("Term", [' 36 months', ' 60 months'])
        home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE'])
        purpose = st.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other'])
        verification_status = st.selectbox("Verification Status", ['Not Verified', 'Verified', 'Source Verified'])
        total_pymnt = st.number_input("Total Payment Received ($)", min_value=0.0, max_value=40000.0, value=0.0)
        recoveries = st.number_input("Recoveries ($)", min_value=0.0, max_value=10000.0, value=0.0)
        out_prncp = st.number_input("Outstanding Principal ($)", min_value=0.0, max_value=40000.0, value=loan_amnt)
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare PD input
        pd_input = pd.DataFrame({
            'loan_amnt': [loan_amnt], 'int_rate': [int_rate], 'dti': [dti], 'annual_inc': [annual_inc],
            'revol_bal': [revol_bal], 'revol_util': [revol_util], 'delinq_2yrs': [delinq_2yrs],
            'open_acc': [open_acc], 'total_acc': [total_acc], 'grade': [grade], 'term': [term],
            'home_ownership': [home_ownership], 'purpose': [purpose], 'verification_status': [verification_status]
        })
        # Placeholder for WoE (simplified)
        woe_cols = [f'{feat}_woe' for feat in pd_features]
        pd_input_woe = pd.DataFrame(0, index=[0], columns=woe_cols)
        pd_input_scaled = scaler.transform(pd_input_woe)

        # Prepare LGD/EAD input
        lgd_ead_input = pd.DataFrame({
            'loan_amnt': [loan_amnt], 'total_pymnt': [total_pymnt], 'recoveries': [recoveries],
            'out_prncp': [out_prncp], 'int_rate': [int_rate], 'dti': [dti], 'grade': [grade],
            'term': [term], 'home_ownership': [home_ownership]
        })
        lgd_ead_input = pd.get_dummies(lgd_ead_input, columns=['grade', 'term', 'home_ownership'], drop_first=True)
        for col in X_lgd_ead.columns:
            if col not in lgd_ead_input.columns:
                lgd_ead_input[col] = 0
        lgd_ead_input = lgd_ead_input[X_lgd_ead.columns]
        lgd_ead_scaled = scaler.transform(lgd_ead_input)

        # Predict
        pd = pd_model.predict_proba(pd_input_scaled)[:, 1][0]
        lgd = lgd_model.predict(lgd_ead_scaled)[0].clip(0, 1)
        ead = ead_model.predict(lgd_ead_scaled)[0].clip(0)
        el = pd * lgd * ead

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Probability of Default (PD)**: {pd:.2%}")
        st.write(f"**Loss Given Default (LGD)**: {lgd:.2%}")
        st.write(f"**Exposure at Default (EAD)**: ${ead:,.2f}")
        st.write(f"**Expected Loss (EL)**: ${el:,.2f}")

        # SHAP explanation
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(pd_input_scaled)
        st.subheader("SHAP Explanation")
        shap.summary_plot(shap_values, pd_input_woe, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        st.image(buf, caption="SHAP Summary Plot")

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    st.write("Upload a CSV file with loan data to predict PD, LGD, EAD, and EL for multiple loans.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        batch_data['id'] = range(1, len(batch_data) + 1)

        # Prepare PD input
        batch_pd = batch_data[pd_features].copy()
        batch_pd_woe = pd.DataFrame(0, index=batch_pd.index, columns=[f'{feat}_woe' for feat in pd_features])
        batch_pd_scaled = scaler.transform(batch_pd_woe)

        # Prepare LGD/EAD input
        batch_lgd_ead = batch_data[lgd_ead_features].copy()
        batch_lgd_ead = pd.get_dummies(batch_lgd_ead, columns=['grade', 'term', 'home_ownership'], drop_first=True)
        for col in X_lgd_ead.columns:
            if col not in batch_lgd_ead.columns:
                batch_lgd_ead[col] = 0
        batch_lgd_ead = batch_lgd_ead[X_lgd_ead.columns]
        batch_lgd_ead_scaled = scaler.transform(batch_lgd_ead)

        # Predict
        batch_data['PD'] = pd_model.predict_proba(batch_pd_scaled)[:, 1]
        batch_data['LGD'] = lgd_model.predict(batch_lgd_ead_scaled).clip(0, 1)
        batch_data['EAD'] = ead_model.predict(batch_lgd_ead_scaled).clip(0)
        batch_data['EL'] = batch_data['PD'] * batch_data['LGD'] * batch_data['EAD']

        # Display results
        st.subheader("Batch Prediction Results")
        st.dataframe(batch_data[['id', 'PD', 'LGD', 'EAD', 'EL']])
        
        # Download results
        csv = batch_data[['id', 'PD', 'LGD', 'EAD', 'EL']].to_csv(index=False)
        st.download_button("Download Results", csv, "batch_results.csv", "text/csv")

elif page == "Visualizations":
    st.header("Model Visualizations")
    st.write("Explore model performance and feature importance.")

    viz_options = ["WoE Plots", "IV Bar Plot", "ROC Curve", "Confusion Matrix", "Feature Importance", 
                   "LGD Scatter", "EL Distribution", "SHAP Summary"]
    selected_viz = st.multiselect("Select Visualizations", viz_options, default=viz_options[:3])

    for viz in selected_viz:
        st.subheader(viz)
        img = Image.open(f"{viz.lower().replace(' ', '_')}.png")
        st.image(img, caption=viz)

elif page == "Portfolio Analysis":
    st.header("Portfolio Analysis")
    st.write("Summarize portfolio risk based on the cleaned dataset.")
    
    portfolio_data = pd.read_csv("el_results.csv")
    st.subheader("Portfolio Summary")
    st.write(f"**Total Loans**: {len(portfolio_data)}")
    st.write(f"**Portfolio Expected Loss (EL)**: ${portfolio_data['EL'].sum():,.2f}")
    st.write(f"**Average PD**: {portfolio_data['PD'].mean():.2%}")
    st.write(f"**Average LGD**: {portfolio_data['LGD'].mean():.2%}")
    st.write(f"**Average EAD**: ${portfolio_data['EAD'].mean():,.2f}")

    # EL Distribution
    st.subheader("EL Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(portfolio_data['EL'], bins=50, kde=True)
    plt.title('Distribution of Expected Loss (EL)')
    plt.xlabel('EL ($)')
    st.pyplot(plt)
    plt.close()