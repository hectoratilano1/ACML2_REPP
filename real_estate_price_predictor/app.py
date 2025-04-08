import streamlit as st
import pandas as pd
from src.model import load_model, predict

st.title("🏡 Real Estate Price Predictor")

st.write("Enter property features:")

# ✅ Match names to trained model's features
sqft = st.number_input("Area (sqft)", min_value=100)
beds = st.number_input("Number of Beds", min_value=1, max_value=10)
baths = st.number_input("Number of Baths", min_value=1, max_value=10)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025)
lot_size = st.number_input("Lot Size (sqft)", min_value=100)
basement = st.selectbox("Has Basement?", ["No", "Yes"])
basement = 1 if basement == "Yes" else 0
property_type_Bunglow = st.selectbox("Is it a Bunglow?", ["No", "Yes"])
property_type_Bunglow = 1 if property_type_Bunglow == "Yes" else 0
property_type_Condo = st.selectbox("Is it a Condo?", ["No", "Yes"])
property_type_Condo = 1 if property_type_Condo == "Yes" else 0

# Auto-generated or fixed fields (you may want to collect from user too)
year_sold = 2025
property_tax = 1200
insurance = 1500
popular = 1
recession = 0
property_age = year_sold - year_built

# ✅ Match model's feature names exactly
input_dict = {
    "year_sold": year_sold,
    "property_tax": property_tax,
    "insurance": insurance,
    "beds": beds,
    "baths": baths,
    "sqft": sqft,
    "year_built": year_built,
    "lot_size": lot_size,
    "basement": basement,
    "popular": popular,
    "recession": recession,
    "property_age": property_age,
    "property_type_Bunglow": property_type_Bunglow,
    "property_type_Condo": property_type_Condo
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Price"):
    model = load_model()

    # Align features
    expected_features = list(model.feature_names_in_)
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_features]

    # Debug info
   # st.subheader("🧪 DEBUGGING INFO")
   # st.write("✅ Expected model features:")
   # st.write(expected_features)
   # st.write("📦 Input DataFrame sent to model:")
   # st.dataframe(input_df)

    prediction = predict(model, input_df)[0]
    st.success(f"💰 Estimated Price: ${prediction:,.2f}")
