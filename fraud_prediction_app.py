import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡️",
    layout="centered"
)

# Title & Description
st.title("🛡️ Transaction Fraud Detection App")
st.write("Enter the transaction details below to check if it is fraud using your trained **XGBoost** model.")

# ---------------------------------------------------------
# Load Saved Model and Scaler
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    # Load trained XGBoost model
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load fitted MinMaxScaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

try:
    model, scaler = load_assets()
    st.success("Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Please make sure 'xgb_model.pkl' and 'scaler.pkl' are saved in the same directory as this file.")
    st.stop()

# Same order LabelEncoder used in your notebook (alphabetical): CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Exact column order the scaler and model were trained on
FEATURES = ["hour", "night", "orig_empty", "diff_orig", "diff_dest",
            "error_orig", "error_dest", "log_Amount", "Type"]

# ---------------------------------------------------------
# User Input Interface
# ---------------------------------------------------------
st.subheader("Input Transaction Details")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (hour since start, 1 - 743)", min_value=1, max_value=743, value=12, step=1)
    txn_type = st.selectbox("Transaction Type", options=TYPES, index=1)
    amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=1000.0, format="%.2f")

with col2:
    old_orig = st.number_input("Sender balance BEFORE", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    new_orig = st.number_input("Sender balance AFTER", min_value=0.0, value=40000.0, step=1000.0, format="%.2f")
    old_dest = st.number_input("Receiver balance BEFORE", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    new_dest = st.number_input("Receiver balance AFTER", min_value=0.0, value=10000.0, step=1000.0, format="%.2f")

threshold = st.slider("Fraud threshold (lower = catches more fraud, more false alarms)", 0.05, 0.95, 0.50, 0.05)

# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
st.markdown("---")

if st.button("🚀 Check Transaction", use_container_width=True):
    # 1. Build the 9 features exactly like your FRAUD_visualization notebook
    hour = step % 24
    features = pd.DataFrame([{
        "hour": hour,
        "night": int(0 <= hour <= 8),
        "orig_empty": int(new_orig == 0),
        "diff_orig": old_orig - new_orig,
        "diff_dest": new_dest - old_dest,
        "error_orig": old_orig - amount - new_orig,
        "error_dest": old_dest + amount - new_dest,
        "log_Amount": np.log1p(amount),
        "Type": TYPES.index(txn_type),
    }])[FEATURES]

    # 2. Scale features using the loaded scaler
    scaled_inputs = scaler.transform(features)

    # 3. Predict fraud probability using loaded XGBoost model
    fraud_prob = float(model.predict_proba(scaled_inputs)[0, 1])

    # 4. Display result
    st.metric(label="Fraud Probability", value=f"{fraud_prob:.1%}")
    st.progress(min(fraud_prob, 1.0))

    if fraud_prob >= threshold:
        st.error("🚨 FRAUD suspected! Please review this transaction.")
    else:
        st.success("✅ Looks like a NORMAL transaction.")