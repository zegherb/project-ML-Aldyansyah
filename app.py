import streamlit as st
import pandas as pd
import pickle

# Load semua model
ridge = pickle.load(open('ridge.pkl', 'rb'))
dt = pickle.load(open('dt.pkl', 'rb'))
poly_model = pickle.load(open('poly.pkl', 'rb'))
poly_transform = pickle.load(open('poly_transform.pkl', 'rb'))

columns = pickle.load(open('columns.pkl', 'rb'))

st.title("Prediksi Gaji Profesional")

# PILIH MODEL
model_choice = st.selectbox("Pilih Model", [
    "Ridge Regression",
    "Decision Tree",
    "Polynomial Regression"
])

# INPUT
experience = st.number_input("Experience Years", 0, 50, 1)
skills = st.number_input("Skills Count", 0, 50, 1)
certifications = st.number_input("Certifications", 0, 20, 0)

education = st.selectbox("Education Level", 
    ["High School", "Diploma", "Bachelor", "Master", "PhD"])

edu_mapping = {'High School': 1, 'Diploma': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
education = edu_mapping[education]

# INPUT DICTIONARY
input_dict = {
    'experience_years': experience,
    'skills_count': skills,
    'certifications': certifications,
    'education_level': education
}

# Isi dummy kolom lain
for col in columns:
    if col not in input_dict:
        input_dict[col] = 0

input_df = pd.DataFrame([input_dict])
input_df = input_df[columns]

# PREDIKSI
if st.button("Prediksi"):

    if model_choice == "Ridge Regression":
        pred = ridge.predict(input_df)

    elif model_choice == "Decision Tree":
        pred = dt.predict(input_df)

    elif model_choice == "Polynomial Regression":
        data_poly = poly_transform.transform(input_df)
        pred = poly_model.predict(data_poly)

    st.success(f"Prediksi Gaji: {int(pred[0])}")