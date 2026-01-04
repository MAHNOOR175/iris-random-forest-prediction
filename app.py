import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("random_forest_iris.pkl")

# Class names
class_names = ["Setosa", "Versicolor", "Virginica"]

# App title
st.set_page_config(page_title="Iris Flower Prediction", layout="centered")
st.title("🌸 Iris Flower Species Prediction")
st.write("Enter the flower measurements below to predict the species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    st.success(f"🌼 Predicted Species: **{class_names[prediction]}**")

    st.subheader("Prediction Probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction_proba[0][i]:.2f}")
