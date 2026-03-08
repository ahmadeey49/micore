import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pdfplumber
import matplotlib.pyplot as plt
import re

# App Configuration
st.set_page_config(page_title="AI-Kudi Score MVP", layout="centered", page_icon="💰")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #28a745; color: white; font-weight: bold; }
    .stNumberInput, .stFileUploader { margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("💰 AI-Kudi Score")
st.write("### AI-powered Financial Inclusion for Micro-Entrepreneurs")
st.info("This system leverages Machine Learning to provide credit scoring for small business owners, bridging the gap for the unbanked population.")

# ====== 1. AI Model Training (Mock Data) ======
@st.cache_resource
def train_ai_model():
    # Training the model with synthetic financial data
    data = pd.DataFrame({
        'income': [20000, 45000, 15000, 60000, 10000, 80000, 25000, 5000, 100000, 12000],
        'expenses': [5000, 25000, 12000, 30000, 8000, 20000, 15000, 4500, 40000, 11000],
        'existing_loans': [0, 1, 0, 2, 0, 1, 0, 1, 0, 2],
        'target': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # 1 = Eligible, 0 = Not Eligible
    })
    X = data[['income', 'expenses', 'existing_loans']]
    y = data['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_ai_model()

# ====== 2. User Interface Selection ======
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Select Applicant Type:", ["Unbanked (Manual Input)", "Banked (Digital Statement)"])

income, expenses, loans = 0, 0, 0

if option == "Banked (Digital Statement)":
    st.subheader("📁 Upload Bank Statement (PDF)")
    uploaded_file = st.file_uploader("Upload your recent bank statement for AI analysis", type=['pdf'])
    
    if uploaded_file:
        with st.spinner('AI is analyzing the statement...'):
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                
                # Extracting numerical values (Financial Data)
                numbers = [int(n) for n in re.findall(r'\b\d{3,10}\b', text.replace(',', ''))]
                if len(numbers) >= 2:
                    income = max(numbers)  
                    expenses = sum(numbers) // len(numbers)  
                st.success(f"Analysis Complete: Detected Income (N{income:,.2f}), Estimated Expenses (N{expenses:,.2f})")
            except:
                st.error("Error reading PDF. Please ensure it is a valid document or enter data manually.")
    
    loans = st.number_input("Number of Active Loans:", min_value=0, step=1)

else:
    st.subheader("✍️ Manual Business Data Entry")
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Average Monthly Income (N):", min_value=0, value=25000)
    with col2:
        expenses = st.number_input("Average Monthly Expenses (N):", min_value=0, value=12000)
    loans = st.number_input("Number of Active Loans:", min_value=0, step=1)

# ====== 3. Prediction & Visuals ======
if st.button("Generate AI Credit Score"):
    input_df = pd.DataFrame([[income, expenses, loans]], columns=['income', 'expenses', 'existing_loans'])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction] * 100

    st.divider()
    
    # Visualizing Cash Flow
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['Monthly Expenses', 'Monthly Income'], [expenses, income], color=['#dc3545', '#28a745'])
    ax.set_ylabel("Amount (Naira)")
    ax.set_title("Financial Health Overview")
    st.pyplot(fig)

    if prediction == 1:
        st.balloons()
        st.success(f"### RESULT: ELIGIBLE FOR CREDIT 🎉")
        st.metric(label="AI Confidence Score", value=f"{prob:.1f}%")
        st.write("This applicant shows a healthy financial ratio and low risk.")
    else:
        st.error(f"### RESULT: NOT ELIGIBLE AT THIS TIME ❌")
        st.metric(label="AI Confidence Score", value=f"{prob:.1f}%")
        st.write("Recommendation: Reduce operational expenses or increase revenue to improve the score.")

# ====== Credits Section ======
st.sidebar.write("---")
st.sidebar.markdown(f"""
*Built by:* Abba Saminu  
*Course:* AI/ML (Foundational)  
*Project:* NextGen Knowledge Showcase  
*Pillar:* Financial Inclusion  
*Date:* March 2026
""")
