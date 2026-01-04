import streamlit as st
import numpy as np
import joblib

model = joblib.load("cancer_model.pkl")

st.set_page_config(page_title="Cancer Prediction")
st.title("🩺 Breast Cancer Prediction")

mean_radius = st.number_input("Mean Radius", 5.0, 30.0, 14.0)
mean_texture = st.number_input("Mean Texture", 5.0, 40.0, 20.0)
mean_perimeter = st.number_input("Mean Perimeter", 40.0, 200.0, 90.0)
mean_area = st.number_input("Mean Area", 100.0, 2500.0, 600.0)

if st.button("Predict"):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area]])
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.error("🔴 Malignant (Cancer Detected)")
    else:
        st.success("🟢 Benign (No Cancer)")
