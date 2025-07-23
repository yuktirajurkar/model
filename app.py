import streamlit as st
import joblib
import numpy as np
import json

# Load model and columns
model = joblib.load('titanic_model.pkl')
with open("model_columns.json") as f:
    model_columns = json.load(f)

st.title("Titanic Survival Predictor")

# User input
sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 100, 25)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encoding inputs
sex = 0 if sex == "male" else 1
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Combine features
family_size = sibsp + parch + 1

input_data = np.array([[
    pclass, sex, age, sibsp, parch, fare,
    embarked_C, embarked_Q, embarked_S, family_size
]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did not survive"
    st.success(f"The passenger would have: {result}")
