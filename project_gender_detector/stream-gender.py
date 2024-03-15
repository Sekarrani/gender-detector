# LIBRARY YANG DIGUNAKAN
import pandas as pd
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as pt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

st.markdown("""
    <style>
    .centered-title {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# JUDUL WEBSITE
# Menampilkan judul dengan class "centered-title"
st.markdown('<h1 class="centered-title">Deteksi Gender</h1>',
            unsafe_allow_html=True)

# membaca model
gender_model = pd.read_csv("gender.csv")

# Bagi data menjadi fitur (X_data) dan label (y_data)
X_data = gender_model.drop("gender", axis=1)
y_data = gender_model["gender"]

# Bagi data menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42)

# MENU OPSI
menu_option = st.selectbox('Menu', ['Form Input', 'Upload File'])
st.write('')
st.write('')
st.write('')
st.write('')

# HALAMAN FORM INPUT
if menu_option == 'Form Input':
    # Form
    st.header('Input Single Data')
    long_hair = (st.radio('Input nilai Panjang Rambut', ['Pendek', 'Panjang']))
    if long_hair == 'Pendek':
        long_hair = 0
    elif long_hair == 'Panjang':
        long_hair = 1
    st.write("")

    forehead_width_cm = st.text_input('Input nilai Lebar Dahi dalam cm')
    deskripsiForeHeadWidth = '   *Cara input nilai form ini: Lebar Dahi dalam cm (5 - 30)'
    st.text(deskripsiForeHeadWidth)
    if forehead_width_cm != '':
        if float(forehead_width_cm) > 30 and float(forehead_width_cm) < 5:
            st.warning('Terlalu besar')
    st.write("")

    forehead_height_cm = st.text_input('Input Nilai Tinggi Dahi dalam cm')
    deskripsiForeHeadHeight = '   *Cara input nilai form ini: Tinggi Dahi dalam cm (5 - 25)'
    st.text(deskripsiForeHeadHeight)
    if forehead_height_cm != '':
        if float(forehead_height_cm) > 25 and float(forehead_height_cm) < 5:
            st.warning('Terlalu besar')
    st.write("")

    nose_wide = (st.radio('Input Nilai Lebar Hidung',['Tidak Lebar', 'Lebar']))
    if nose_wide == 'Tidak Lebar':
        nose_wide = 0
    elif nose_wide == 'Lebar':
        nose_wide = 1
    st.write("")

    nose_long = (st.radio('Input Nilai Panjang Hidung', ['Pendek', 'Panjang']))
    if nose_long == 'Pendek':
        nose_long = 0
    elif nose_long == 'Panjang':
        nose_long = 1
    st.write("")

    lips_thin = (st.radio('Input Nilai Ketebalan Bibir', ['Tipis', 'Tebal']))
    if lips_thin == 'Tipis':
        lips_thin = 0
    elif lips_thin == 'Tebal':
        lips_thin = 1
    st.write("")

    distance_nose_to_lip_long = (st.radio('Input Nilai Jarak Antara Hidung dengan Bibir', ['Pendek', 'Panjang']))
    if distance_nose_to_lip_long == 'Pendek':
        distance_nose_to_lip_long = 0
    elif distance_nose_to_lip_long == 'Panjang':
        distance_nose_to_lip_long = 1
    st.write("")

    # Train the Gaussian Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Button to trigger classification
    if st.button('Test Klasifikasi Gender'):
        # Perform prediction on the test data
        if forehead_width_cm != '' and float(forehead_width_cm) <= 30 and forehead_height_cm != '' and float(forehead_height_cm) <= 25 and float(forehead_height_cm) >= 5 and float(forehead_width_cm) >= 5:
            
            # PREDIKSI
            X_test_prediction = gnb.predict([[long_hair, float(forehead_width_cm), float(
                forehead_height_cm), nose_wide, nose_long, lips_thin, distance_nose_to_lip_long]])

            # MENAMPILKAN HASIL PREDIKSI
            st.success(X_test_prediction[0])
        else:
            # AKAN ADA PERINGATAN BILA ADA DATA YANG BELUM KE INPUT
            st.warning(
                'Pastikan nilai Lebar Dahi dan Tinggi Dahi diisi dan tidak lebih dari 30 dan 25, juga input dengan bilangan asli')

# UPLOAD FILE ->
elif menu_option == 'Upload File':
    
    # UPLOAD FILE
    st.header('Upload File')
    uploaded_file = st.file_uploader("Upload file CSV")

    # MEMBACA / MENYIMPAN FILE DALAM VARIABEL
    if uploaded_file is not None:
        gender_model_file = pd.read_csv(uploaded_file)

    # MENAMPILKAN DATASET
    st.write()
    st.header('Tampilan Dataset')
    if uploaded_file is not None:
        st.dataframe(gender_model_file)

    # INFORMASI DATASET FILE
    st.header('Informasi Dataset')
    st.write('Ukuran Tabel')
    st.write(gender_model_file.shape)
    st.write('Tipe Data Tabel Setiap Kolom')
    st.write(gender_model_file.dtypes)

    # MENGUBAH NILAI DARI KOLOM GENDER MENJADI BINER
    gender_model_file.gender = [1 if i == "Male" else 0 for i in gender_model_file.gender]

    # TAMPILAN DATASET BARU SETELAH DI UBAH KOLOM GENDER
    st.write()
    st.header('Tampilan Dataset Terbaru')
    gender_model_file

    # MEMISAHKAN DATA X DAN DATA Y
    if uploaded_file is not None:
        # x_data
        x_data = gender_model_file.drop(["gender"], axis=1)
        # y_data[]
        y_data = gender_model_file.gender.values

    # STANDARISASI DATA -> ditransform data yang berskala, biar seimbang
    scaler = StandardScaler()
    scaler.fit(x_data)
    Standarized_data = scaler.transform(x_data)

    x_data = Standarized_data
    y_data = gender_model_file['gender']

    # INPUT ANGKA PERSENTASE DATA TESTING YANG AKAN DIGUNAKAN
    st.header('Input Persentase Data Testing')
    test_input = st.number_input('Persentase Data Testing', 0.4, 1.0, step=0.1)
    test_size = float(test_input)

    # SPLIT DATA / MEMISAHKAN DATA TRAINING DAN TESTING
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, stratify=y_data, random_state=2, shuffle=True)
    
    # JUMLAH DATA TRAINING DAN TESTING
    st.header('Jumlah Data Testing ')
    st.write('(jumlah data testing, jumlah kolom)')
    st.write(x_test.shape)

    # MELAKUKAN KLASIFIKASI DENGAN NBC
    gnb = GaussianNB()

    gnb.fit(x_train, y_train)

# BILA BUTTON KLASIFIKASI DITEKAN, MENAMPILKAN DATA TRAINING DAN TESTING
    # Button to trigger classification
    if st.button('Test Klasifikasi Gender Dengan File'):
        x_train_prediction = gnb.predict(x_train)
        training_data_accuracy = accuracy_score(x_train_prediction, y_train)

        st.header('Menampilkan Hasil Akurasi')
        st.write('Akurasi data training = ', training_data_accuracy)

        x_test_prediction = gnb.predict(x_test)
        test_data_accuracy = accuracy_score(x_test_prediction, y_test)

        st.write('Akurasi data testing = ', test_data_accuracy)
