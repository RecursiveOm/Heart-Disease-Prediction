import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>by Dr. Omkar Avinash Zunje</p>", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

with col2:
    sex = st.selectbox("Sex", ['M', 'F'])
    chest_pain = st.selectbox("Chest Pain", ["ATA", "NAP", "TA", "ASY"])
    fasting_bs = st.selectbox("Fasting BS > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

if st.button("🔍 Predict", use_container_width=True):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    proba = model.predict_proba(scaled_input)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High Risk ({round(proba*100,2)}%)")
    else:
        st.success(f"✅ Low Risk ({round(proba*100,2)}%)")