import streamlit as st
import pandas as pd
import pickle

# --- Page configuration ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# --- Custom Styling ---
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border-radius: 8px;
}
div.stButton > button:first-child:hover {
    background-color: #ff3333;
}
</style>
""", unsafe_allow_html=True)

# --- Load the trained model ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error("Model not found! Please run `train.py` first to train and save the model.")
    st.stop()

# --- Main App ---
st.title("📉 Customer Churn Prediction System")
st.markdown("""
Welcome to the internal tool for identifying at-risk customers. 
Enter the customer's demographics, service subscriptions, and billing details to predict if they are going to leave the company ("Churn").
""")

# --- Form Layout ---
with st.container():
    st.header("1. Billing Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input('Tenure (Months)', min_value=0, max_value=100, value=12, help="How many months the customer has been with the company")
    with col2:
        MonthlyCharges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=50.0)
    with col3:
        TotalCharges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=float(tenure * 50.0))

st.markdown("---")

with st.container():
    st.header("2. Customer Demographics")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
    with col_d2:
        SeniorCitizen_str = st.selectbox('Senior Citizen', ['No', 'Yes'])
        SeniorCitizen = 1 if SeniorCitizen_str == 'Yes' else 0
    with col_d3:
        Partner = st.selectbox('Has Partner', ['Yes', 'No'])
    with col_d4:
        Dependents = st.selectbox('Has Dependents', ['Yes', 'No'])

st.markdown("---")

with st.container():
    st.header("3. Subscribed Services")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
        MultipleLines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    with col_s2:
        InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    with col_s3:
        OnlineBackup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        DeviceProtection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])

    col_s4, col_s5, col_s6 = st.columns(3)
    with col_s4:
        TechSupport = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    with col_s5:
        StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    with col_s6:
        StreamingMovies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

st.markdown("---")

with st.container():
    st.header("4. Contract & Account Details")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    with col_a2:
        PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
    with col_a3:
        PaymentMethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

st.markdown("<br>", unsafe_allow_html=True)

# --- Prediction Action ---
if st.button("🚀 Predict Churn Risk"):
    # Group inputs into a dictionary
    input_dict = {
        'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner], 'Dependents': [Dependents],
        'tenure': [tenure], 'PhoneService': [PhoneService], 'MultipleLines': [MultipleLines],
        'InternetService': [InternetService], 'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport], 'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies], 'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges]
    }
    
    # Create DataFrame to match model schema
    input_data = pd.DataFrame(input_dict)
    
    # Predict using the loaded model
    with st.spinner('Analyzing profile...'):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    st.subheader("💡 Prediction Result")
    
    if prediction_proba < 0.3:
        st.success(f"🟢 **Low Risk!** This customer is highly likely to stay.")
    elif prediction_proba < 0.7:
        st.warning(f"🟡 **Medium Risk!** This customer might leave. Consider promotional offers.")
    else:
        st.error(f"🔴 **High Risk!** This customer is highly likely to leave. Immediate action required.")
        
    st.write(f"**Probability of Churning:** {prediction_proba * 100:.2f}%")
    st.progress(float(prediction_proba))
