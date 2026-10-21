"""
app.py
Streamlit Web Application for Telecom Customer Churn Prediction and Retention Intelligence.
Built upon the end-to-end, leakage-aware ML pipeline from Customer_Churn_.ipynb.
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import model pipeline builder and helpers
from model_pipeline import (
    load_or_train_pipeline,
    train_and_export_pipeline,
    generate_benchmark_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_ECONOMICS,
)

# Page configuration
st.set_page_config(
    page_title="TeleChurn AI - Customer Churn & Retention Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .high-risk {
        color: #DC2626;
    }
    .med-risk {
        color: #D97706;
    }
    .low-risk {
        color: #16A34A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def get_model_bundle():
    """Load or train pipeline bundle once and cache."""
    return load_or_train_pipeline()


try:
    bundle = get_model_bundle()
    pipeline = bundle["pipeline"]
    notebook_threshold = bundle.get("threshold", 0.317)
    economics = bundle.get("economics", DEFAULT_ECONOMICS)
    model_name = bundle.get("model_name", "Gradient Boosting (Calibrated)")
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=64)
    st.title("TeleChurn Intelligence")
    st.caption("AI-Powered Churn Risk & Retention Optimizer")

    st.divider()

    st.subheader("⚙️ Decision Engine Config")
    st.markdown(f"**Model:** `{model_name}`")

    threshold_mode = st.radio(
        "Decision Threshold Strategy",
        ["Optimized F1 (Notebook: 0.317)", "Standard (0.500)", "Custom Slider"],
        help="The notebook selected ~0.317 based on training out-of-fold calibration to maximize F1 under class imbalance.",
    )

    if threshold_mode == "Optimized F1 (Notebook: 0.317)":
        active_threshold = notebook_threshold
    elif threshold_mode == "Standard (0.500)":
        active_threshold = 0.500
    else:
        active_threshold = st.slider("Custom Threshold", 0.05, 0.95, float(notebook_threshold), 0.01)

    st.info(f"🎯 **Active Threshold:** `{active_threshold:.3f}`\n\nCustomers with probability ≥ {active_threshold:.3f} are flagged for retention.")

    st.divider()

    st.subheader("💼 Campaign Economics (INR ₹)")
    contact_cost = st.number_input("Contact Cost per Customer (₹)", value=float(economics.get("contact_cost_inr", 100.0)), step=10.0)
    monthly_contribution = st.number_input("Monthly Margin Contribution (₹)", value=float(economics.get("monthly_contribution_inr", 500.0)), step=50.0)
    months_saved = st.number_input("Months of Value Retained", value=int(economics.get("months_of_value_saved", 6)), step=1)
    save_prob = st.slider("Campaign Save Rate", 0.05, 0.80, float(economics.get("campaign_save_probability", 0.25)), 0.05)
    capacity = st.number_input("Weekly Contact Capacity", value=int(economics.get("weekly_contact_capacity", 500)), step=50)

    value_if_saved = monthly_contribution * months_saved
    breakeven_p = (contact_cost / (save_prob * value_if_saved)) if (save_prob * value_if_saved) > 0 else 1.0

    st.caption(f"💡 **Economic Breakeven Risk:** `{breakeven_p:.1%}` probability. Targeting customers with churn risk above this is expected to yield positive net ROI.")

    st.divider()
    if st.button("🔄 Retrain / Refresh Model Pipeline"):
        with st.spinner("Retraining calibrated pipeline..."):
            bundle = train_and_export_pipeline()
            st.cache_resource.clear()
            st.success("Pipeline retrained & cached!")
            st.rerun()


# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-title">📡 Telecom Customer Churn & Retention Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Leakage-Aware Machine Learning System | Probability Calibration & Economic Optimization</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Individual Customer Scoring",
    "📁 Batch Cohort Scoring & Export",
    "💰 Campaign ROI & Economic Simulator",
    "📊 Model Performance & Diagnostics",
])


# ==========================================
# TAB 1: INDIVIDUAL CUSTOMER SCORING
# ==========================================
with tab1:
    st.subheader("Interactive Customer Risk Assessment")
    st.markdown("Select a sample persona or configure individual account parameters to evaluate real-time churn risk and prescribe retention offers.")

    # Persona Buttons
    col_preset, col_clear = st.columns([4, 1])
    with col_preset:
        preset = st.selectbox(
            "⚡ Quick Load Profile:",
            [
                "Custom Manual Input",
                "🚨 High-Risk Profile (New Month-to-Month, Fiber Optic, Electronic Check)",
                "🛡️ Low-Risk Loyal Profile (Long Tenure, 2-Year Contract, Tech Support)",
                "⚖️ Moderate-Risk Profile (1-Year Contract, Streaming Bundles)",
            ],
        )

    # Defaults setup based on persona
    if "High-Risk" in preset:
        d_gender = "Female"
        d_senior = 1
        d_partner = "No"
        d_dependents = "No"
        d_tenure = 2
        d_phone = "Yes"
        d_mult_lines = "Yes"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_bkp = "No"
        d_prot = "No"
        d_supp = "No"
        d_tv = "Yes"
        d_mov = "Yes"
        d_contract = "Month-to-month"
        d_paperless = "Yes"
        d_payment = "Electronic check"
        d_monthly = 95.50
        d_total = 191.00
    elif "Low-Risk" in preset:
        d_gender = "Male"
        d_senior = 0
        d_partner = "Yes"
        d_dependents = "Yes"
        d_tenure = 60
        d_phone = "Yes"
        d_mult_lines = "Yes"
        d_internet = "DSL"
        d_sec = "Yes"
        d_bkp = "Yes"
        d_prot = "Yes"
        d_supp = "Yes"
        d_tv = "No"
        d_mov = "No"
        d_contract = "Two year"
        d_paperless = "No"
        d_payment = "Credit card (automatic)"
        d_monthly = 65.00
        d_total = 3900.00
    elif "Moderate-Risk" in preset:
        d_gender = "Female"
        d_senior = 0
        d_partner = "Yes"
        d_dependents = "No"
        d_tenure = 24
        d_phone = "Yes"
        d_mult_lines = "No"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_bkp = "Yes"
        d_prot = "No"
        d_supp = "No"
        d_tv = "Yes"
        d_mov = "Yes"
        d_contract = "One year"
        d_paperless = "Yes"
        d_payment = "Bank transfer (automatic)"
        d_monthly = 85.00
        d_total = 2040.00
    else:
        d_gender = "Male"
        d_senior = 0
        d_partner = "No"
        d_dependents = "No"
        d_tenure = 12
        d_phone = "Yes"
        d_mult_lines = "No"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_bkp = "No"
        d_prot = "No"
        d_supp = "No"
        d_tv = "No"
        d_mov = "No"
        d_contract = "Month-to-month"
        d_paperless = "Yes"
        d_payment = "Electronic check"
        d_monthly = 70.00
        d_total = 840.00

    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 1. Demographic & Household")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if d_gender == "Female" else 1)
            senior = st.selectbox("Senior Citizen", [0, 1], index=d_senior, format_func=lambda x: "Yes (65+)" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"], index=0 if d_partner == "Yes" else 1)
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if d_dependents == "Yes" else 1)

            st.markdown("#### 2. Account & Contract")
            tenure = st.slider("Tenure with Telco (months)", 1, 72, int(d_tenure))
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(d_contract),
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if d_paperless == "Yes" else 1)
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ].index(d_payment),
            )

        with col2:
            st.markdown("#### 3. Subscribed Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if d_phone == "Yes" else 1)
            mult_lines_opts = ["No phone service", "No", "Yes"] if phone == "No" else ["No", "Yes"]
            mult_lines = st.selectbox(
                "Multiple Lines",
                mult_lines_opts,
                index=0 if d_mult_lines not in mult_lines_opts else mult_lines_opts.index(d_mult_lines),
            )

            internet = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(d_internet),
            )

            inet_sec_opts = ["No internet service"] if internet == "No" else ["No", "Yes"]
            online_sec = st.selectbox("Online Security", inet_sec_opts, index=0 if d_sec not in inet_sec_opts else inet_sec_opts.index(d_sec))
            online_bkp = st.selectbox("Online Backup", inet_sec_opts, index=0 if d_bkp not in inet_sec_opts else inet_sec_opts.index(d_bkp))
            device_prot = st.selectbox("Device Protection", inet_sec_opts, index=0 if d_prot not in inet_sec_opts else inet_sec_opts.index(d_prot))
            tech_supp = st.selectbox("Tech Support", inet_sec_opts, index=0 if d_supp not in inet_sec_opts else inet_sec_opts.index(d_supp))

        with col3:
            st.markdown("#### 4. Media & Streaming")
            stream_tv = st.selectbox("Streaming TV", inet_sec_opts, index=0 if d_tv not in inet_sec_opts else inet_sec_opts.index(d_tv))
            stream_mov = st.selectbox("Streaming Movies", inet_sec_opts, index=0 if d_mov not in inet_sec_opts else inet_sec_opts.index(d_mov))

            st.markdown("#### 5. Financial Charges")
            monthly = st.number_input("Monthly Charges (₹ / $)", min_value=15.0, max_value=150.0, value=float(d_monthly), step=1.0)
            auto_total = round(monthly * tenure, 2)
            total = st.number_input(
                "Total Charges Accumulated",
                min_value=15.0,
                max_value=12000.0,
                value=float(d_total if d_total > 0 else auto_total),
                step=10.0,
                help="Historical cumulative charges billed to this customer.",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("⚡ Predict Customer Churn Risk", use_container_width=True)

    # Prediction Logic
    customer_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": mult_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bkp,
        "DeviceProtection": device_prot,
        "TechSupport": tech_supp,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    prob = float(pipeline.predict_proba(customer_df)[:, 1][0])
    is_churn_pred = prob >= active_threshold

    # Calculate expected value for this customer
    expected_ret_val = prob * save_prob * value_if_saved
    expected_net_val = expected_ret_val - contact_cost

    st.markdown("---")
    st.markdown("### 📋 Prediction Results & Prescription")

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        risk_class = "high-risk" if prob >= 0.50 else ("med-risk" if prob >= active_threshold else "low-risk")
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 0.9rem; color: #64748B;">CALIBRATED CHURN PROBABILITY</span>
            <div class="metric-value {risk_class}">{prob:.1%}</div>
            <span style="font-size: 0.8rem; color: #64748B;">Threshold: {active_threshold:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        flag_label = "FLAGGED FOR OUTREACH" if is_churn_pred else "RETAINED / HEALTHY"
        flag_color = "#DC2626" if is_churn_pred else "#16A34A"
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 0.9rem; color: #64748B;">ACTION STATUS</span>
            <div class="metric-value" style="color: {flag_color}; font-size: 1.25rem; margin-top: 10px;">{flag_label}</div>
            <span style="font-size: 0.8rem; color: #64748B;">Risk Tier: {"High" if prob >= 0.50 else ("Moderate" if prob >= active_threshold else "Low")}</span>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        val_color = "#16A34A" if expected_net_val > 0 else "#DC2626"
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 0.9rem; color: #64748B;">EXPECTED NET VALUE (ROI)</span>
            <div class="metric-value" style="color: {val_color}; font-size: 1.5rem;">₹{expected_net_val:,.1f}</div>
            <span style="font-size: 0.8rem; color: #64748B;">Gross Save: ₹{expected_ret_val:,.1f}</span>
        </div>
        """, unsafe_allow_html=True)

    with res_col4:
        econ_rec = "✅ Positive ROI Outreach" if expected_net_val > 0 else "❌ Negative ROI (Do Not Contact)"
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 0.9rem; color: #64748B;">ECONOMIC RECOMMENDATION</span>
            <div class="metric-value" style="font-size: 1.15rem; margin-top: 10px; color: #1E293B;">{econ_rec}</div>
            <span style="font-size: 0.8rem; color: #64748B;">Contact Cost: ₹{contact_cost:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)

    # Prescriptive Playbook & Explanations
    st.markdown("<br>", unsafe_allow_html=True)
    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.markdown("#### 🔍 Primary Risk Factor Breakdown")
        factors = []
        if contract == "Month-to-month":
            factors.append(("Month-to-Month Contract", "High Risk Driver (+0.07 importance)", "Customers without commitment churn at >42% rate."))
        if internet == "Fiber optic":
            factors.append(("Fiber Optic Internet", "Elevated Churn (+0.03 importance)", "Fiber users churn higher due to price sensitivity & technical support tickets."))
        if tech_supp == "No":
            factors.append(("No Tech Support", "Lack of Attachment", "Users without tech support churn 3x faster when issues occur."))
        if online_sec == "No":
            factors.append(("No Online Security", "Security Vulnerability", "Value-add services create platform stickiness."))
        if payment == "Electronic check":
            factors.append(("Electronic Check Payment", "Frictional Billing", "Manual monthly payments invite active re-evaluation of bills."))
        if tenure <= 6:
            factors.append(("Low Tenure (<= 6 months)", "Onboarding Phase", "Customer has not yet formed long-term retention habits."))

        if factors:
            for title, badge, desc in factors:
                st.warning(f"**{title}** — *{badge}*\n\n{desc}")
        else:
            st.success("🎉 **No major risk flags detected.** Customer exhibits high loyalty indicators (Long tenure, multi-year contract, bundled services).")

    with p_col2:
        st.markdown("#### 🎯 Prescriptive Retention Playbook")
        actions = []
        if contract == "Month-to-month":
            actions.append("📝 **Offer 1-Year or 2-Year Contract Discount**: Providing a 10-15% monthly discount in exchange for a contract commitment neutralizes the #1 churn driver.")
        if tech_supp == "No" or online_sec == "No":
            actions.append("🛡️ **Complimentary Security & Tech Support Bundle**: Offer 3 months of free dedicated VIP Tech Support and Security suite.")
        if payment == "Electronic check":
            actions.append("💳 **Autopay Incentive**: Offer a one-time ₹250 / $10 bill credit to switch from Electronic Check to Automatic Bank Transfer or Credit Card.")
        if tenure <= 6:
            actions.append("🤝 **Proactive CS Outreach**: Schedule a concierge check-in call to address speed or service satisfaction within first 90 days.")

        if not actions:
            actions.append("⭐ **Loyalty Reward**: Send gratitude perk or device upgrade eligibility notification to maintain high NPS.")

        for act in actions:
            st.info(act)


# ==========================================
# TAB 2: BATCH CSV SCORING
# ==========================================
with tab2:
    st.subheader("Cohort Scoring & Batch Export")
    st.markdown("Upload customer records in CSV format or score our built-in benchmark cohort to prioritize outreach at scale.")

    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
    with b_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("📊 Load Sample Cohort (200 Customers)", use_container_width=True)

    df_to_score = None
    if uploaded_file is not None:
        try:
            df_to_score = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}` ({len(df_to_score):,} rows)")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_sample:
        df_to_score = generate_benchmark_data(n_samples=200, seed=123)
        st.info("Loaded sample Telco cohort of 200 customers.")

    if df_to_score is not None:
        scoring_data = df_to_score.copy()
        drop_cols = [c for c in ["Churn", "customerID"] if c in scoring_data.columns]
        X_score = scoring_data.drop(columns=drop_cols)

        try:
            probs = pipeline.predict_proba(X_score)[:, 1]
            scored_df = df_to_score.copy()
            scored_df["churn_probability"] = np.round(probs, 4)
            scored_df["predicted_churn"] = (probs >= active_threshold).astype(int)
            scored_df["risk_tier"] = np.where(
                probs >= 0.50, "High Risk", np.where(probs >= active_threshold, "Moderate Risk", "Low Risk")
            )
            scored_df["expected_gross_saved_inr"] = np.round(probs * save_prob * value_if_saved, 2)
            scored_df["expected_net_value_inr"] = np.round(scored_df["expected_gross_saved_inr"] - contact_cost, 2)
            scored_df["recommended_for_outreach"] = (scored_df["expected_net_value_inr"] > 0) & (scored_df["predicted_churn"] == 1)

            scored_df = scored_df.sort_values("expected_net_value_inr", ascending=False).reset_index(drop=True)

            total_customers = len(scored_df)
            flagged = int((scored_df["predicted_churn"] == 1).sum())
            profitable_targets = int((scored_df["expected_net_value_inr"] > 0).sum())
            total_potential_net = float(scored_df[scored_df["expected_net_value_inr"] > 0]["expected_net_value_inr"].sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Cohort Size", f"{total_customers:,}")
            m2.metric("Flagged for Churn", f"{flagged:,} ({flagged/total_customers:.1%})")
            m3.metric("Economically Viable Targets", f"{profitable_targets:,}")
            m4.metric("Potential Net Value", f"₹{total_potential_net:,.0f}")

            st.markdown("#### Scored Customer Queue (Ranked by Expected Net Value)")
            display_cols = [c for c in [
                "customerID", "churn_probability", "risk_tier", "predicted_churn",
                "expected_net_value_inr", "Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod"
            ] if c in scored_df.columns]

            st.dataframe(scored_df[display_cols].head(50), use_container_width=True)

            csv_data = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Complete Scored Dataset (CSV)",
                data=csv_data,
                file_name="telecom_churn_scored_customers.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error scoring dataset: {e}. Please ensure input columns match the standard Telco schema.")


# ==========================================
# TAB 3: CAMPAIGN ROI SIMULATOR
# ==========================================
with tab3:
    st.subheader("Retention Campaign Financial ROI Simulator")
    st.markdown("Model executive scenarios and operational capacity constraints using the decision model from Section 9 of the notebook.")

    s_col1, s_col2 = st.columns([1, 1])

    with s_col1:
        st.markdown("#### 🎯 Capacity & Unit Economics")
        sim_cohort_size = st.slider("Customer Base Being Monitored", 1000, 50000, 10000, 1000)
        sim_churn_base = st.slider("Baseline Cohort Churn Rate", 0.10, 0.40, 0.265, 0.005)

        st.write(f"- **Cost per Contact**: ₹{contact_cost:,.0f}")
        st.write(f"- **Contribution Margin per Month**: ₹{monthly_contribution:,.0f}")
        st.write(f"- **Customer Value if Saved (6 mo)**: ₹{value_if_saved:,.0f}")
        st.write(f"- **Campaign Save Probability**: {save_prob:.0%}")
        st.write(f"- **Operational Contact Capacity**: {capacity:,} contacts / cycle")

    with s_col2:
        st.markdown("#### 📊 Projected Campaign Outcomes")

        sim_probs = np.random.beta(a=1.5, b=4.5, size=sim_cohort_size)
        sim_gross = sim_probs * save_prob * value_if_saved
        sim_net = sim_gross - contact_cost

        eligible = sim_net > 0
        n_eligible = int(eligible.sum())
        n_selected = min(int(capacity), n_eligible)

        top_indices = np.argsort(sim_net)[::-1][:n_selected]
        campaign_gross = float(sim_gross[top_indices].sum())
        campaign_cost = float(n_selected * contact_cost)
        campaign_net = campaign_gross - campaign_cost
        campaign_roi = (campaign_net / campaign_cost * 100) if campaign_cost > 0 else 0.0

        r1, r2 = st.columns(2)
        r1.metric("Eligible Accounts (ROI > 0)", f"{n_eligible:,}")
        r2.metric("Targeted within Capacity", f"{n_selected:,}")

        r3, r4 = st.columns(2)
        r3.metric("Expected Net Profit Added", f"₹{campaign_net:,.0f}")
        r4.metric("Campaign Net ROI", f"{campaign_roi:.1f}%")

    st.markdown("#### 📈 Cumulative Net Return vs. Contacted Volume")
    sorted_net = np.sort(sim_net)[::-1]
    cum_net = np.cumsum(sorted_net)
    contacts_axis = np.arange(1, len(cum_net) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(contacts_axis[:min(2500, len(cum_net))], cum_net[:min(2500, len(cum_net))], color="#1E3A8A", lw=2, label="Cumulative Net Value (₹)")
    ax.axvline(x=capacity, color="#DC2626", linestyle="--", label=f"Weekly Capacity Limit ({capacity:,})")
    ax.axvline(x=n_eligible, color="#16A34A", linestyle=":", label=f"Economic Break-Even Volume ({n_eligible:,})")
    ax.set_xlabel("Number of Contacted Customers (Ranked by Risk/Value)")
    ax.set_ylabel("Cumulative Net Value (₹)")
    ax.set_title("Retention Value Optimization Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


# ==========================================
# TAB 4: MODEL DIAGNOSTICS & ANALYTICS
# ==========================================
with tab4:
    st.subheader("Model Performance & Feature Interpretability")
    st.markdown("Detailed cross-validation benchmarks and permutation feature importances extracted from the training notebook.")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("#### 🏆 5-Fold Cross-Validation Model Comparison")
        cv_summary = pd.DataFrame([
            {"Model": "Gradient Boosting (Calibrated)", "Average Precision (PR-AUC)": 0.669, "ROC-AUC": 0.850, "F1-Score": 0.584, "Accuracy": 0.805, "Status": "Selected 🥇"},
            {"Model": "CatBoost Classifier", "Average Precision (PR-AUC)": 0.666, "ROC-AUC": 0.848, "F1-Score": 0.590, "Accuracy": 0.806, "Status": "Candidate"},
            {"Model": "XGBoost Classifier", "Average Precision (PR-AUC)": 0.662, "ROC-AUC": 0.845, "F1-Score": 0.584, "Accuracy": 0.802, "Status": "Candidate"},
            {"Model": "Logistic Regression (Balanced)", "Average Precision (PR-AUC)": 0.660, "ROC-AUC": 0.846, "F1-Score": 0.628, "Accuracy": 0.748, "Status": "Baseline"},
            {"Model": "LightGBM Classifier", "Average Precision (PR-AUC)": 0.645, "ROC-AUC": 0.836, "F1-Score": 0.624, "Accuracy": 0.769, "Status": "Candidate"},
            {"Model": "Random Forest (Balanced)", "Average Precision (PR-AUC)": 0.638, "ROC-AUC": 0.837, "F1-Score": 0.615, "Accuracy": 0.788, "Status": "Candidate"},
            {"Model": "Support Vector Machine (RBF)", "Average Precision (PR-AUC)": 0.607, "ROC-AUC": 0.827, "F1-Score": 0.624, "Accuracy": 0.750, "Status": "Candidate"},
        ])
        st.dataframe(cv_summary.set_index("Model"), use_container_width=True)

        st.markdown("#### 🎯 Threshold Impact on Held-Out Test Set (1,409 records)")
        threshold_table = pd.DataFrame([
            {"Decision Rule": "Standard Threshold (0.50)", "Accuracy": "80.3%", "Precision": "66.2%", "Recall": "52.4%", "F1": "0.585", "PR-AUC": "0.664", "Brier Score": "0.135"},
            {"Decision Rule": "Tuned Threshold (0.317)", "Accuracy": "76.8%", "Precision": "54.5%", "Recall": "75.7%", "F1": "0.634", "PR-AUC": "0.664", "Brier Score": "0.135"},
        ])
        st.dataframe(threshold_table.set_index("Decision Rule"), use_container_width=True)
        st.caption("Notice how lowering the threshold from 0.50 to 0.317 increases churn recall from **52.4% to 75.7%**, catching significantly more churners.")

    with m_col2:
        st.markdown("#### 🌲 Test-Set Permutation Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature": [
                "tenure", "Contract", "InternetService", "TechSupport",
                "OnlineSecurity", "TotalCharges", "MonthlyCharges",
                "PaperlessBilling", "PaymentMethod", "MultipleLines",
                "OnlineBackup", "SeniorCitizen", "StreamingMovies",
                "Dependents", "gender"
            ],
            "Importance": [
                0.1258, 0.0697, 0.0338, 0.0237,
                0.0157, 0.0144, 0.0137,
                0.0063, 0.0049, 0.0049,
                0.0027, 0.0015, 0.0010,
                0.0005, 0.0000
            ]
        }).sort_values("Importance", ascending=True)

        fig2, ax2 = plt.subplots(figsize=(8, 6.5))
        bars = ax2.barh(feat_imp["Feature"], feat_imp["Importance"], color="#3B82F6")
        ax2.set_xlabel("Mean Test-Set Metric Deterioration (PR-AUC)")
        ax2.set_title("Permutation Feature Importance (Predictive Drivers)")
        ax2.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)


# Footer
st.markdown("---")
st.caption("TeleChurn AI Suite | Powered by Streamlit, Scikit-Learn, and CalibratedClassifierCV | Designed for Enterprise Retention Teams")
