
import streamlit as st
import pandas as pd
from src.model import load_model, predict

st.title("🏡 Real Estate Price Predictor")

st.write("Enter property features:")

# Example input features (adjust as per dataset)
area = st.number_input("Area (sqft)", min_value=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10)
floors = st.selectbox("Number of Floors", [1, 2, 3])
property_type_Bunglow = st.selectbox("Is Bunglow?", ["No", "Yes"])
property_type_Bunglow = 1 if property_type_Bunglow == "Yes" else 0

# Modify input features based on your dataset
input_dict = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "floors": floors,
    "property_type_Bunglow": property_type_Bunglow
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Price"):
    model = load_model()
    prediction = predict(model, input_df)[0]
    st.success(f"💰 Estimated Price: ${prediction:,.2f}")
