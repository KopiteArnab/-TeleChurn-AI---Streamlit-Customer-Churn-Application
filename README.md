# 📡 TeleChurn AI - Streamlit Customer Churn Application

An end-to-end, production-ready Streamlit web application for telecom customer churn prediction and retention economics, built on the leakage-aware machine learning pipeline from `Customer_Churn_.ipynb`.

---

## 🚀 Quickstart: Running the App

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
Run this command from this directory:
```bash
streamlit run app.py
```
The application will automatically launch in your browser at `http://localhost:8501`.

---

## 🌟 Key Features

1. **👤 Individual Customer Risk Scorer**:
   - Interactive form covering Demographics, Subscribed Services, Account & Contract, and Billing.
   - Quick-load persona profiles: *🚨 High-Risk*, *🛡️ Loyal Long-Term*, *⚖️ Moderate-Risk*.
   - Probability Gauge with Risk Tiers (*Low*, *Moderate*, *High*).
   - Prescriptive Next-Best-Action recommendations based on the top churn drivers.

2. **📁 Cohort & Batch CSV Scoring**:
   - Upload any customer CSV or click **Load Sample Cohort (200 Customers)**.
   - Computes churn probability, decision flags, and expected net savings per account.
   - Download the full scored dataset with risk flags in CSV format.

3. **💰 Retention ROI & Campaign Simulator**:
   - Direct implementation of the economic optimization model from the notebook.
   - Configurable parameters: Contact Cost, Monthly Contribution Margin, Retained Months, and Campaign Save Probability.
   - Break-even risk calculation and Capacity-constrained volume optimization.

4. **📊 Model Performance & Feature Analytics**:
   - 5-fold cross-validation leaderboard comparing Gradient Boosting, CatBoost, XGBoost, Logistic Regression, and Random Forest.
   - Permutation feature importances ranking the top churn drivers (*tenure*, *Contract*, *InternetService*, *TechSupport*, etc.).
   - Decision threshold comparison: standard 0.50 vs. tuned F1 threshold (0.317).
