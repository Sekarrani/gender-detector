import pickle as p
import streamlit as st

# membaca model
gender_model = p.load(open('gender.csv'))

# judul web 
st.title('Data Mining Gender')