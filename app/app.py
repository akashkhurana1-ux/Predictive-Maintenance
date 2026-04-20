import streamlit as st
import joblib
import pandas as pd

model = joblib.load("../model/model.pkl")

st.title("Engine Failure Prediction")

rpm = st.number_input("Engine RPM")
oil_pressure = st.number_input("Lub Oil Pressure")
fuel_pressure = st.number_input("Fuel Pressure")
coolant_pressure = st.number_input("Coolant Pressure")
oil_temp = st.number_input("Lub Oil Temperature")
coolant_temp = st.number_input("Coolant Temperature")

input_df = pd.DataFrame([[rpm, oil_pressure, fuel_pressure, coolant_pressure, oil_temp, coolant_temp]],
columns=[
'Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure',
'Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature'
])

if st.button("Predict"):
    pred = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    st.write("Prediction:", pred[0])
    st.write("Failure Probability:", prob)
