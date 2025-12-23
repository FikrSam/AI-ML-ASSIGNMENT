import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Analysis", layout="wide")

# Load all saved artifacts
try:
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    metrics = joblib.load('model_results.pkl')
except FileNotFoundError:
    st.error("Missing model files. Please run 'model_training.py' first.")
    st.stop()

st.title("House Price Prediction Dashboard")
st.write("Real-time property valuation based on comparative Machine Learning analysis.")

# --- Algorithm Comparison Section (DYNAMIC) ---
st.header("1. Algorithm Comparison")
st.markdown("Metrics updated based on the latest training session:")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Linear Regression")
    st.write("---")
    st.metric("R-squared (R²)", f"{metrics['lr_r2']:.4f}")
    st.write(f"**Mean Absolute Error:** ${metrics['lr_mae']:,.2f}")
    if metrics['lr_r2'] >= metrics['rf_r2']:
        st.success("Verdict: Best Performing Model")

with col_b:
    st.subheader("Random Forest")
    st.write("---")
    st.metric("R-squared (R²)", f"{metrics['rf_r2']:.4f}")
    st.write(f"**Mean Absolute Error:** ${metrics['rf_mae']:,.2f}")
    if metrics['rf_r2'] > metrics['lr_r2']:
        st.success("Verdict: Best Performing Model")
    else:
        st.info("Verdict: Secondary Model")

st.divider()

# --- Prediction UI ---
st.header("2. Property Valuation Tool")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Total Area (sq ft)", min_value=500, max_value=20000, value=5000)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 1)
        stories = st.slider("Stories", 1, 4, 1)
        parking = st.slider("Parking Spaces", 0, 3, 1)
    with col2:
        mainroad = st.selectbox("Main Road Access", ["Yes", "No"])
        guestroom = st.selectbox("Guestroom", ["Yes", "No"])
        basement = st.selectbox("Basement", ["Yes", "No"])
        hotwater = st.selectbox("Hot Water Heating", ["Yes", "No"])
        aircon = st.selectbox("Air Conditioning", ["Yes", "No"])
        prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
        furnishing = st.selectbox("Furnishing Status", ["Semi-furnished", "Furnished", "Unfurnished"])

    submit = st.form_submit_button("Predict Price")

if submit:
    input_data = {
        'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'stories': stories, 'mainroad': 1 if mainroad == "Yes" else 0,
        'guestroom': 1 if guestroom == "Yes" else 0, 'basement': 1 if basement == "Yes" else 0,
        'hotwaterheating': 1 if hotwater == "Yes" else 0, 'airconditioning': 1 if aircon == "Yes" else 0,
        'parking': parking, 'prefarea': 1 if prefarea == "Yes" else 0,
        'furnishingstatus_semi-furnished': 1 if furnishing == "Semi-furnished" else 0,
        'furnishingstatus_unfurnished': 1 if furnishing == "Unfurnished" else 0
    }

    input_df = pd.DataFrame([input_data]).reindex(columns=feature_columns, fill_value=0)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.success(f"Estimated Market Price: ${prediction:,.2f}")
