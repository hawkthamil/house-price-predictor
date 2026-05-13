import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load
model = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")

st.title("🏠 House Price Predictor (Clean Model)")

st.write("Enter house details:")

LotArea = st.number_input("Lot Area", 1000, 20000, 5000)
YearBuilt = st.number_input("Year Built", 1900, 2025, 2000)
OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.number_input("Living Area (sqft)", 500, 5000, 1500)
TotalBsmtSF = st.number_input("Basement Area", 0, 3000, 800)
GarageCars = st.slider("Garage Cars", 0, 4, 2)

input_data = pd.DataFrame([[LotArea, YearBuilt, OverallQual, GrLivArea, TotalBsmtSF, GarageCars]],
                          columns=features)

if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")