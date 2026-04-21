import streamlit as st
import pandas as pd
import pickle

# Load model & kolom
model = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("Prediksi Gaji Profesional")

# INPUT USER
experience = st.number_input("Experience Years", 0, 50, 1)
skills = st.number_input("Skills Count", 0, 50, 1)
certifications = st.number_input("Certifications", 0, 20, 0)

education = st.selectbox("Education Level", 
    ["High School", "Diploma", "Bachelor", "Master", "PhD"])

# Mapping education
edu_mapping = {'High School': 1, 'Diploma': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
education = edu_mapping[education]

# Buat dataframe input
input_dict = {
    'experience_years': experience,
    'skills_count': skills,
    'certifications': certifications,
    'education_level': education
}

# Tambahin semua kolom dummy = 0
for col in columns:
    if col not in input_dict:
        input_dict[col] = 0

# Urutkan sesuai training
input_df = pd.DataFrame([input_dict])
input_df = input_df[columns]

# PREDIKSI
if st.button("Prediksi"):
    pred = model.predict(input_df)
    st.success(f"Prediksi Gaji: {int(pred[0])}")