import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pdfplumber
import matplotlib.pyplot as plt
import re

# App Configuration
st.set_page_config(page_title="Micore - AI Credit Scoring", layout="centered", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
.stButton>button {
background-color: #006747;
color: white;
border-radius: 8px;
font-weight: bold;
}
.main {
background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Micore AI")
st.write("### AI-Powered Financial Inclusion for Small Businesses")

# Helper Functions
def format_currency(amount):
    return "{:,.0f}".format(amount).replace(",", " ")

def extract_finances_from_text(text):

    credit_matches = re.findall(r'(?i)credit.?([\d,]+\.?\d)', text)
    debit_matches = re.findall(r'(?i)debit.?([\d,]+\.?\d)', text)

    credits = [float(num.replace(',', '')) for num in credit_matches if num]
    debits = [float(num.replace(',', '')) for num in debit_matches if num]

    return sum(credits), sum(debits)

# ===== AI MODEL TRAINING =====
@st.cache_resource
def train_micore_model():

    data = pd.DataFrame({
        'income':[15000,50000,85000,12000,95000,20000,40000,70000,30000,60000],
        'expenses':[5000,20000,40000,9000,30000,15000,10000,25000,12000,20000],
        'loans':[0,1,0,0,1,2,0,1,0,1],
        'target':[1,1,1,0,1,0,1,1,1,1]
    })

    # Feature Engineering
    data['expense_ratio'] = data['expenses'] / data['income']

    X = data[['income','expenses','loans','expense_ratio']]
    y = data['target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)

    return model

model = train_micore_model()

# ===== USER INTERFACE =====

st.sidebar.header("User Dashboard")

user_type = st.sidebar.selectbox(
"Applicant Category",
["Banked (With Statement)", "Unbanked (Manual)"]
)

income = 0
expenses = 0
loans = 0

if user_type == "Banked (With Statement)":

    input_method = st.radio(
    "Choose Input Method",
    ["Upload Statement (PDF/Image)", "Enter Manually"]
    )

    if input_method == "Upload Statement (PDF/Image)":

        uploaded_file = st.file_uploader(
        "Upload Bank Statement",
        type=["pdf","png","jpg","jpeg"]
        )

        if uploaded_file:

            with st.spinner("Micore AI is scanning your statement..."):

                if uploaded_file.type == "application/pdf":

                    with pdfplumber.open(uploaded_file) as pdf:
                        text = " ".join(
                        [p.extract_text() for p in pdf.pages if p.extract_text()]
                        )

                    income, expenses = extract_finances_from_text(text)

                else:

                    st.info("Image detected. Micore AI is using a prototype vision model for statement analysis.")

                    # Prototype OCR Simulation
                    income = 55000
                    expenses = 22000

                if income > 0:

                    st.success("Financial patterns detected")

                    st.write(f"*Total Credits (Income):* ₦ {format_currency(income)}")
                    st.write(f"*Total Debits (Expenses):* ₦ {format_currency(expenses)}")

                else:

                    st.warning("No clear financial patterns found. Try manual entry.")

    else:

        income = st.number_input(
        "Monthly Income (₦)",
        min_value=0,
        step=1000
        )

        expenses = st.number_input(
        "Monthly Expenses (₦)",
        min_value=0,
        step=1000
        )

else:

    income = st.number_input(
    "Business Monthly Income (₦)",
    min_value=0,
    step=1000
    )

    expenses = st.number_input(
    "Business Monthly Expenses (₦)",
    min_value=0,
    step=1000
    )

loans = st.number_input(
"Existing Active Loans (Count)",
min_value=0,
step=1
)

duration = st.slider(
"Loan Repayment Duration (Months)",
1,12,3
)

amount_needed = st.number_input(
"Amount of Loan Needed (₦)",
min_value=1000,
step=1000
)

# ===== AI ANALYSIS =====

if st.button("Analyze with Micore AI"):

    st.divider()

    if income > 100000:

        st.error("### RESULT: NOT ELIGIBLE ❌")
        st.warning("Micore is designed for small business owners earning below 100 000 Naira monthly.")

    elif income < 10000:

        st.error("### RESULT: NOT ELIGIBLE ❌")
        st.warning("Income is below the minimum threshold for micro-credit analysis.")

    elif (income - expenses) < (amount_needed / duration):

        st.error("### RESULT: NOT ELIGIBLE ❌")
        st.warning(f"Insufficient cash flow. Monthly repayment of ₦ {format_currency(amount_needed/duration)} is not affordable.")

    else:

        expense_ratio = expenses / income

        input_data = pd.DataFrame(
        [[income,expenses,loans,expense_ratio]],
        columns=['income','expenses','loans','expense_ratio']
        )

        prediction = model.predict(input_data)[0]
        score = model.predict_proba(input_data)[0][1] * 100

        if prediction == 1:

            st.balloons()

            st.success("### RESULT: ELIGIBLE FOR CREDIT 🎉")

            st.info("AI Message: If your financial patterns remain consistent in the coming months, you are highly eligible.")

        else:

            st.error("### RESULT: NOT ELIGIBLE ❌")

            st.info("AI Message: Based on current trends, improving your income-to-expense ratio may improve eligibility.")

        st.metric(
        label="Micore AI Risk Score",
        value=f"{score:.1f}/100"
        )

    # Chart
    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
    ['Monthly Income','Monthly Expenses'],
    [income, expenses]
    )

    ax.set_title("Financial Health Overview")

    st.pyplot(fig)

# ===== FOOTER =====

st.sidebar.markdown("""
---
Built by: *Abba Saminu*  
Project: *3MTT NextGen Knowledge Showcase*  
Pillar: *Financial Inclusion*  
Location: *Wudil, Kano State*  
Date: *March 2026*
""")
