# Install dependencies: pip install pandas numpy scikit-learn imblearn xgboost shap matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, r2_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# 1. Data Processing
# Load data with low_memory=False to avoid DtypeWarning
data = pd.read_csv('loan.csv', low_memory=False)  # Replace with your file path

# Add imaginary IDs
data['id'] = range(1, len(data) + 1)

# Drop irrelevant or problematic columns to avoid mixed types
drop_cols = ['member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code', 'funded_amnt', 
             'funded_amnt_inv', 'pymnt_plan', 'initial_list_status', 'policy_code', 
             'application_type', 'next_pymnt_d', 'hardship_reason', 'hardship_status', 
             'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date', 
             'settlement_status', 'settlement_date', 'verification_status_joint', 
             'revol_bal_joint', 'sec_app_earliest_cr_line']
data = data.drop(columns=[col for col in drop_cols if col in data.columns])

# Handle missing values
numeric_cols = ['loan_amnt', 'int_rate', 'dti', 'annual_inc', 'revol_bal', 'total_pymnt', 
                'out_prncp', 'revol_util', 'delinq_2yrs', 'open_acc', 'total_acc']
categorical_cols = ['grade', 'home_ownership', 'term', 'purpose', 'verification_status']
for col in numeric_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].mode()[0])

# Convert percentages
data['int_rate'] = data['int_rate'] / 100
if 'revol_util' in data.columns:
    data['revol_util'] = data['revol_util'].apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x)

# Handle outliers
data['annual_inc'] = data['annual_inc'].clip(lower=data['annual_inc'].quantile(0.01),
                                            upper=data['annual_inc'].quantile(0.99))
data['dti'] = data['dti'].clip(lower=0, upper=data['dti'].quantile(0.99))

# Save cleaned data
data.to_csv("cleaned_loans.csv", index=False)

# 2. Data Preparation
# Define target for PD
data['target'] = (data['loan_status'] == 'Charged Off').astype(int)
data = data[data['loan_status'].isin(['Charged Off', 'Fully Paid'])]

# Select features
pd_features = ['loan_amnt', 'int_rate', 'dti', 'annual_inc', 'revol_bal', 'revol_util', 
               'delinq_2yrs', 'open_acc', 'total_acc', 'grade', 'term', 'home_ownership', 
               'purpose', 'verification_status']
lgd_ead_features = ['loan_amnt', 'total_pymnt', 'recoveries', 'out_prncp', 'int_rate', 'dti', 
                    'grade', 'term', 'home_ownership']

# WoE and IV calculation
def calculate_woe_iv(df, feature, target, bins=10):
    df = df[[feature, target]].copy()
    if df[feature].dtype in ['float64', 'int64']:
        df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop', labels=False)
    else:
        df['bin'] = df[feature]
    grouped = df.groupby('bin')[target].agg(['count', 'sum'])
    grouped['non_default'] = grouped['count'] - grouped['sum']
    grouped['non_default_dist'] = grouped['non_default'] / grouped['non_default'].sum()
    grouped['default_dist'] = grouped['sum'] / grouped['sum'].sum()
    grouped['non_default_dist'] = grouped['non_default_dist'].replace(0, 0.0001)
    grouped['default_dist'] = grouped['default_dist'].replace(0, 0.0001)
    grouped['woe'] = np.log(grouped['non_default_dist'] / grouped['default_dist'])
    grouped['iv'] = (grouped['non_default_dist'] - grouped['default_dist']) * grouped['woe']
    iv = grouped['iv'].sum()
    return grouped, iv

# Calculate WoE and IV
woe_dict = {}
iv_dict = {}
for feature in pd_features:
    if feature in data.columns:
        woe_df, iv = calculate_woe_iv(data, feature, 'target')
        woe_dict[feature] = woe_df['woe']
        iv_dict[feature] = iv

iv_df = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['IV']).sort_values(by='IV', ascending=False)
print("Information Value (IV):")
print(iv_df)

# Transform PD features with WoE
X_pd = data[pd_features].copy()
for feature in pd_features:
    if feature in data.columns and feature in woe_dict:
        if X_pd[feature].dtype in ['float64', 'int64']:
            X_pd['bin'] = pd.qcut(X_pd[feature], q=10, duplicates='drop', labels=False)
        else:
            X_pd['bin'] = X_pd[feature]
        X_pd[feature + '_woe'] = X_pd['bin'].map(woe_dict[feature])
        X_pd = X_pd.drop(columns=['bin'])
X_pd = X_pd[[col for col in X_pd.columns if '_woe' in col]]

# Prepare LGD/EAD data
X_lgd_ead = data[lgd_ead_features].copy()
X_lgd_ead = pd.get_dummies(X_lgd_ead, columns=['grade', 'term', 'home_ownership'], drop_first=True)

# Scale features
scaler = StandardScaler()
X_pd_scaled = scaler.fit_transform(X_pd)
X_lgd_ead_scaled = scaler.fit_transform(X_lgd_ead)

# Handle class imbalance for PD
y_pd = data['target']
smote = SMOTE(random_state=42)
X_pd_resampled, y_pd_resampled = smote.fit_resample(X_pd_scaled, y_pd)

# Split data
X_pd_train, X_pd_test, y_pd_train, y_pd_test = train_test_split(X_pd_resampled, y_pd_resampled, 
                                                                test_size=0.2, random_state=42, stratify=y_pd_resampled)
X_lgd_ead_train, X_lgd_ead_test = train_test_split(X_lgd_ead_scaled, test_size=0.2, random_state=42)

# 3. Modeling
# PD (Logistic Regression)
pd_model = LogisticRegression(class_weight='balanced', random_state=42)
pd_model.fit(X_pd_train, y_pd_train)
y_pd_pred_proba = pd_model.predict_proba(X_pd_test)[:, 1]
y_pd_pred = (y_pd_pred_proba >= 0.5).astype(int)
auc = roc_auc_score(y_pd_test, y_pd_pred_proba)
print(f"PD Logistic Regression AUC-ROC: {auc:.2f}")
print("Classification Report:")
print(classification_report(y_pd_test, y_pd_pred))

# LGD (Linear Regression)
defaulted_data = data[data['loan_status'] == 'Charged Off'].copy()
defaulted_data['LGD'] = 1 - (defaulted_data['total_pymnt'] / defaulted_data['loan_amnt'])
X_lgd = X_lgd_ead[defaulted_data.index]
y_lgd = defaulted_data['LGD']
X_lgd_train, X_lgd_test, y_lgd_train, y_lgd_test = train_test_split(X_lgd, y_lgd, test_size=0.2, random_state=42)
lgd_model = LinearRegression()
lgd_model.fit(X_lgd_train, y_lgd_train)
y_lgd_pred = lgd_model.predict(X_lgd_test)
lgd_r2 = r2_score(y_lgd_test, y_lgd_pred)
print(f"LGD Linear Regression R2: {lgd_r2:.2f}")

# EAD (Linear Regression)
defaulted_data['EAD'] = defaulted_data['out_prncp'] if 'out_prncp' in defaulted_data.columns else defaulted_data['loan_amnt']
y_ead = defaulted_data['EAD']
X_ead_train, X_ead_test, y_ead_train, y_ead_test = train_test_split(X_lgd, y_ead, test_size=0.2, random_state=42)
ead_model = LinearRegression()
ead_model.fit(X_ead_train, y_ead_train)
y_ead_pred = ead_model.predict(X_ead_test)
ead_r2 = r2_score(y_ead_test, y_ead_pred)
print(f"EAD Linear Regression R2: {ead_r2:.2f}")

# 4. Expected Loss (EL)
data['PD'] = pd_model.predict_proba(X_pd_scaled)[:, 1]
data['LGD'] = lgd_model.predict(X_lgd_ead_scaled)
data['EAD'] = ead_model.predict(X_lgd_ead_scaled)
data['LGD'] = data['LGD'].clip(0, 1)
data['EAD'] = data['EAD'].clip(0)
data['EL'] = data['PD'] * data['LGD'] * data['EAD']
portfolio_el = data['EL'].sum()
print(f"Portfolio Expected Loss: ${portfolio_el:,.2f}")
data[['id', 'PD', 'LGD', 'EAD', 'EL']].to_csv("el_results.csv", index=False)

# 5. Graphical Interpretations
# WoE Plots
for feature in pd_features[:3]:
    woe_df, _ = calculate_woe_iv(data, feature, 'target')
    plt.figure(figsize=(8, 5))
    sns.barplot(x=woe_df.index, y='woe', data=woe_df)
    plt.title(f'WoE for {feature}')
    plt.xticks(rotation=45)
    plt.savefig(f'woe_plot_{feature}.png')
    plt.close()

# IV Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=iv_df.index, y='IV', data=iv_df)
plt.title('Information Value (IV) by Feature')
plt.xticks(rotation=90)
plt.axhline(y=0.1, color='r', linestyle='--', label='IV=0.1 (Medium)')
plt.legend()
plt.savefig('iv_plot.png')
plt.close()

# ROC Curve (PD)
fpr, tpr, _ = roc_curve(y_pd_test, y_pd_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (PD)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Confusion Matrix (PD)
cm = confusion_matrix(y_pd_test, y_pd_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (PD)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature Importance (PD)
coef_df = pd.DataFrame({'Feature': X_pd.columns, 'Coefficient': pd_model.coef_[0]})
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.sort_values(by='Coefficient', ascending=False))
plt.title('Feature Importance (PD - Logistic Regression)')
plt.savefig('feature_importance.png')
plt.close()

# LGD Prediction vs. Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_lgd_test, y_lgd_pred, alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')
plt.title('LGD: Predicted vs. Actual')
plt.xlabel('Actual LGD')
plt.ylabel('Predicted LGD')
plt.savefig('lgd_scatter.png')
plt.close()

# EL Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['EL'], bins=50, kde=True)
plt.title('Distribution of Expected Loss (EL)')
plt.xlabel('EL ($)')
plt.savefig('el_distribution.png')
plt.close()

# XGBoost with SHAP
xgb_model = xgb.XGBClassifier(scale_pos_weight=len(y_pd_train[y_pd_train==0])/len(y_pd_train[y_pd_train==1]), random_state=42)
xgb_model.fit(X_pd_train, y_pd_train)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_pd_test)
shap.summary_plot(shap_values, X_pd_test, show=False)
plt.savefig('shap_summary.png')
plt.close()