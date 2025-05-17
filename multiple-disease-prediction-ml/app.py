import streamlit as st
import pickle
import numpy as np

st.title("Multiple Disease Prediction System")

disease = st.sidebar.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Parkinson's"])

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

def predict_disease(model, input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Positive" if prediction[0] == 1 else "Negative"

if disease == "Diabetes":
    st.subheader("Diabetes Prediction")
    input_data = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(8)]
    if st.button("Predict"):
        model = load_model("diabetes_model.pkl")
        st.success(f"Prediction: {predict_disease(model, input_data)}")

elif disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    input_data = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(13)]
    if st.button("Predict"):
        model = load_model("heart_model.pkl")
        st.success(f"Prediction: {predict_disease(model, input_data)}")

else:
    st.subheader("Parkinson's Disease Prediction")
    input_data = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(22)]
    if st.button("Predict"):
        model = load_model("parkinsons_model.pkl")
        st.success(f"Prediction: {predict_disease(model, input_data)}")
