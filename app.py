import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = upper)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 500.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

if st.button("Predict"):
    # Encode categorical inputs
    sex = 0 if sex == "male" else 1
    embarked_C = 1 if embarked == "C" else 0
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0

    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked_C': embarked_C,
        'Embarked_Q': embarked_Q,
        'Embarked_S': embarked_S
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # Output
    if prediction == 1:
        st.success("🎉 Survived!")
    else:
        st.error("☠️ Did not survive.")
