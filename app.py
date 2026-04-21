import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))

st.title("Prediksi ML Sederhana")

input_data = st.number_input("Masukkan angka")

if st.button("Prediksi"):
    hasil = model.predict([[input_data]])
    st.write("Hasil:", hasil[0])