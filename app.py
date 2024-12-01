from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import pickle 

model = tf.keras.models.load_model("model1.h5")

with open("LabelEncoder_Gender", "rb") as f:
    label_encoder = pickle.load(f)
    
with open("OneHotEncoder_Geography", "rb") as f:
    onehot_encoder = pickle.load(f)
    
with open("StandardScaler", "rb") as f:
    scaler = pickle.load(f)

## Streamlit app

st.title("Churn Prediction")

## Taking user input
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
credit_score = st.number_input("Credit Score")
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.checkbox("Has Credit Card")
is_active_member = st.checkbox("Is Active Member")

## Prepare the data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],  # Corrected key
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],  # Corrected key
    'EstimatedSalary': [estimated_salary]
}

## OneHotEncoder for Geography
geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out())

df = pd.concat([pd.DataFrame(input_data).reset_index(drop=True), geo_encoded_df], axis=1)

scaled_df = scaler.transform(df)

## Prediction
pred = model.predict(scaled_df)

result = 'Customer is likely to churn' if pred > 0.5 else 'Customer is not likely to churn'

## Highlighted Result 
if result == 'Customer is likely to churn': 
    st.markdown(f"<h2 style='color:red;'>**{result}**</h2>", unsafe_allow_html=True) 
else: 
    st.markdown(f"<h2 style='color:green;'>**{result}**</h2>", unsafe_allow_html=True)