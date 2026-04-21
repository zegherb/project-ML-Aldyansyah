import streamlit as st
import pandas as pd
import pickle
import numpy as np

# LOAD DATASET UNTUK MENGAMBIL DAFTAR UNIK
df_asli = pd.read_csv('job_salary_prediction_dataset.csv')

# LOAD SEMUA MODEL & TRANSFORMER
ridge = pickle.load(open('ridge.pkl', 'rb'))
dt = pickle.load(open('dt.pkl', 'rb'))
poly_model = pickle.load(open('poly.pkl', 'rb'))
poly_transform = pickle.load(open('poly_transform.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb')) 
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("🚀 Prediksi Gaji Profesional")
st.write("Masukkan profil profesional untuk memprediksi estimasi gaji.")

# --- SIDEBAR: PILIH MODEL ---
model_choice = st.sidebar.selectbox("Algoritma Model", [
    "Ridge Regression", "Decision Tree", "Polynomial Regression"
])

# --- INPUT LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Numerik")
    experience = st.number_input("Pengalaman (Tahun)", 0, 50, 1)
    skills = st.number_input("Jumlah Skill", 0, 50, 1)
    certifications = st.number_input("Jumlah Sertifikat", 0, 20, 0)
    
    st.subheader("Pendidikan")
    education = st.selectbox("Tingkat Pendidikan", ["High School", "Diploma", "Bachelor", "Master", "PhD"])
    edu_mapping = {'High School': 1, 'Diploma': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
    education_val = edu_mapping[education]

with col2:
    st.subheader("Detail Pekerjaan")
    job = st.selectbox("Jabatan (Job Title)", sorted(df_asli['job_title'].unique()))
    industry = st.selectbox("Industri", sorted(df_asli['industry'].unique()))
    location = st.selectbox("Lokasi (Negara)", sorted(df_asli['location'].unique()))
    comp_size = st.selectbox("Ukuran Perusahaan", sorted(df_asli['company_size'].unique()))
    remote = st.selectbox("Kerja Remote", sorted(df_asli['remote_work'].unique()))

# --- PROSES INPUT ---
# dictionary awal (Numerik & Ordinal)
input_dict = {
    'experience_years': experience,
    'skills_count': skills,
    'certifications': certifications,
    'education_level': education_val
}

# Logika One-Hot Encoding untuk semua kategori
kategori_terpilih = {
    'job_title': job,
    'industry': industry,
    'company_size': comp_size,
    'location': location,
    'remote_work': remote
}

for label, pilihan in kategori_terpilih.items():
    col_name = f"{label}_{pilihan}"
    if col_name in columns:
        input_dict[col_name] = 1

# Isi kolom yang tidak terpilih dengan 0
for col in columns:
    if col not in input_dict:
        input_dict[col] = 0

# Ubah ke DataFrame dengan urutan kolom yang benar
input_df = pd.DataFrame([input_dict])[columns]

# STANDARDISASI (Scaling) pada kolom numerik
kolom_numerik = ['experience_years', 'skills_count', 'certifications']
input_df[kolom_numerik] = scaler.transform(input_df[kolom_numerik])

# --- PREDIKSI ---
if st.button("Hitung Estimasi Gaji", type="primary"):
    if model_choice == "Ridge Regression":
        pred = ridge.predict(input_df)
    elif model_choice == "Decision Tree":
        pred = dt.predict(input_df)
    elif model_choice == "Polynomial Regression":
        data_poly = poly_transform.transform(input_df)
        pred = poly_model.predict(data_poly)

    hasil = int(pred[0])
    st.balloons()
    st.metric(label="Estimasi Gaji", value=f"${hasil:,}")
    st.info(f"Model yang digunakan: {model_choice}")