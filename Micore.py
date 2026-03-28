import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pdfplumber
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import datetime

# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="Micore AI Credit Scoring",
                   layout="centered",
                   page_icon="📊")

st.markdown("""
<style>
.stButton>button {
background-color:#006747;
color:white;
border-radius:8px;
font-weight:bold;
height:3em;
}
.main {background-color:#f0f2f6;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Micore AI")
st.write("### AI-Powered Micro Credit Scoring for Small Businesses")

# -------------------------
# Session Storage
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Helper Functions
# -------------------------
def format_currency(amount):
    return "{:,.0f}".format(amount)

def extract_finances_from_text(text):
    credit_matches = re.findall(r'(?i)credit.*?([\d,]+)', text)
    debit_matches = re.findall(r'(?i)debit.*?([\d,]+)', text)
    credits = [float(x.replace(",","")) for x in credit_matches]
    debits = [float(x.replace(",","")) for x in debit_matches]
    return sum(credits), sum(debits)

# -------------------------
# Train AI Model
# -------------------------
@st.cache_resource
def train_model():
    data=pd.DataFrame({
        "income":[15000,50000,85000,12000,95000,20000,40000,70000,30000,60000],
        "expenses":[5000,20000,40000,9000,30000,15000,10000,25000,12000,20000],
        "loans":[0,1,0,0,1,2,0,1,0,1],
        "loan_amount":[10000,20000,40000,8000,50000,15000,20000,30000,25000,35000],
        "duration":[3,6,6,2,6,3,4,5,3,4],
        "target":[1,1,1,0,1,0,1,1,1,1]
    })
    data["expense_ratio"]=data["expenses"]/data["income"]
    monthly_repayment=data["loan_amount"]/data["duration"]
    data["repayment_ratio"]=monthly_repayment/data["income"]
    X=data[["income","expenses","loans","expense_ratio","repayment_ratio"]]
    y=data["target"]
    model=RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X,y)
    return model

model=train_model()

# -------------------------
# Sidebar Input
# -------------------------
st.sidebar.header("User Dashboard")
user_type=st.sidebar.selectbox("Applicant Category",
                               ["Banked (With Statement)","Unbanked (Manual)"])

income=0
expenses=0

# -------------------------
# Alternative Data Inputs
# -------------------------
st.subheader("Alternative Business Data")
daily_sales=st.number_input("Average Daily Sales",min_value=0,value=5000)
customers=st.number_input("Customers per Day",min_value=0,value=20)
shop_rent=st.number_input("Monthly Shop Rent",min_value=0,value=10000)

# -------------------------
# User Income/Expenses
# -------------------------
if user_type=="Banked (With Statement)":
    method=st.radio("Input Method",["Upload Statement","Manual Entry"])
    if method=="Upload Statement":
        uploaded_file=st.file_uploader("Upload Bank Statement",type=["pdf","png","jpg","jpeg"])
        if uploaded_file:
            with st.spinner("Analyzing statement..."):
                if uploaded_file.type=="application/pdf":
                    with pdfplumber.open(uploaded_file) as pdf:
                        text=" ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                    income,expenses=extract_finances_from_text(text)
                else:
                    st.info("Prototype image analysis used")
                    income=55000
                    expenses=22000
                st.success("Statement analyzed")
                st.write("Detected Income:", format_currency(income))
                st.write("Detected Expenses:", format_currency(expenses))
    else:
        income=st.number_input("Monthly Income", min_value=10000, max_value=1000000, step=1000, value=25000)
        expenses=st.number_input("Monthly Expenses", min_value=0, step=1000, value=12000)
else:
    income=st.number_input("Business Monthly Income", min_value=10000, max_value=1000000, step=1000, value=25000)
    expenses=st.number_input("Business Monthly Expenses", min_value=0, step=1000, value=12000)

loans=st.number_input("Existing Loans", min_value=0, step=1)
loan_amount=st.number_input("Loan Amount Needed", min_value=1000, step=1000)
duration=st.slider("Repayment Duration (months)",1,12,3)

# -------------------------
# AI Analysis
# -------------------------
if st.button("Analyze with Micore AI"):
    monthly_cashflow=income-expenses
    monthly_repayment=loan_amount/duration

    # Financial Metrics
    col1,col2,col3=st.columns(3)
    col1.metric("Monthly Income",f"₦{format_currency(income)}")
    col2.metric("Monthly Expenses",f"₦{format_currency(expenses)}")
    col3.metric("Monthly Loan Repayment",f"₦{format_currency(monthly_repayment)}")

    # Fraud Detection
    if income>1000000:
        st.warning("⚠ Suspicious income detected")
    if loan_amount>income*10:
        st.warning("⚠ Requested loan far exceeds income")
    if monthly_cashflow < monthly_repayment*0.95:
        st.error("NOT ELIGIBLE ❌")
        st.warning("Loan repayment exceeds safe cashflow")
    else:
        expense_ratio=expenses/income
        repayment_ratio=monthly_repayment/income
        input_data=pd.DataFrame([[income,expenses,loans,expense_ratio,repayment_ratio]],
                                columns=["income","expenses","loans","expense_ratio","repayment_ratio"])
        prediction=model.predict(input_data)[0]
        score=model.predict_proba(input_data)[0][1]*100

        # Credit Result
        st.subheader("Credit Result")
        if prediction==1:
            st.success("ELIGIBLE FOR CREDIT 🎉")
            st.balloons()
        else:
            st.error("NOT ELIGIBLE ❌")
        st.metric("Micore AI Risk Score", f"{score:.1f}/100")

        # Smart Loan Recommendation
        recommended_loan=income*2
        if loan_amount>recommended_loan:
            st.info(f"Recommended safer loan amount: ₦{format_currency(recommended_loan)}")

        # Business Growth Prediction
        future_income=income*1.2
        st.write(f"📈 If this financial pattern continues, business income may reach ₦{format_currency(future_income)} in 6 months.")

        # Credit Score Gauge
        gauge=go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text":"Micore AI Credit Score"},
            gauge={
                "axis":{"range":[0,100]},
                "steps":[
                    {"range":[0,40],"color":"red"},
                    {"range":[40,70],"color":"orange"},
                    {"range":[70,100],"color":"lightgreen"}
                ]
            }
        ))
        st.plotly_chart(gauge,use_container_width=True)

        # Save History
        st.session_state.history.append({
            "Income":income,
            "Loan":loan_amount,
            "Score":score,
            "Decision":"Approved" if prediction==1 else "Rejected"
        })

        # Financial Overview Chart
        fig,ax=plt.subplots()
        ax.bar(["Income","Expenses"],[income,expenses])
        ax.set_title("Financial Overview")
        st.pyplot(fig)

# -------------------------
# Fintech Dashboard
# -------------------------
if len(st.session_state.history)>0:
    st.subheader("Micore Fintech Analytics")
    df=pd.DataFrame(st.session_state.history)
    total=len(df)
    approved=len(df[df["Decision"]=="Approved"])
    rejected=len(df[df["Decision"]=="Rejected"])
    avg_score=df["Score"].mean()
    col1,col2,col3,col4=st.columns(4)
    col1.metric("Total Applicants",total)
    col2.metric("Approved",approved)
    col3.metric("Rejected",rejected)
    col4.metric("Average Score",f"{avg_score:.1f}")
    st.dataframe(df)
    st.bar_chart(df["Score"])

# -------------------------
# Footer
# -------------------------
st.sidebar.markdown("""
---
Built by: *Abba Saminu*  
Project: *3MTT NextGen Knowledge Showcase*  
Pillar: *Financial Inclusion*  
Location: *Wudil, Kano State*  
Date: *March 2026*
""")
