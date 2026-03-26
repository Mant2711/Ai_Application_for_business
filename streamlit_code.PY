import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model")
imputer = joblib.load("imputer")
scaler = joblib.load("scaler")

st.set_page_config(page_title ="CHD_Prediction" )#page_icon=""
st.title("Coronary_Heart_Disease_prediction")
st.write("Enter_patient_details_here")

age = st.number_input("age", min_value = 5, max_value = 150)
sex = st.selectbox("sex",[0,1], format_func=lambda x:"Female" if x==0 else "Male")
cp = st.number_input("Chest_pain_type", min_value=0, max_value=5, value = 1)
chol = st.number_input("Cholesterol",min_value = 50, max_value = 700)# value = 100
trestbps = st.number_input("Blood_Pressure", min_value = 50, max_value =400 )#value = 120
fbs = st.selectbox("Blood_suger > 120" , [0,1])
thalach = st.number_input("Maximum_Heart_rate", min_value = 50, max_value = 300)#value = 150
exang = st.selectbox("Induced_anigma",[0,1])
oldpeak = st.number_input("oldpeak", min_value = 0.0, max_value = 15.0, step = 0.1)
ca = st.number_input("number_of_vessel", min_value = 0, max_value = 4)
thal = st.number_input("thal", min_value = 0, max_value = 4)#value = 2
restecg = st.number_input("ECG", min_value=0 , max_value=2, value=1)
slope = st.number_input("slope", min_value=0, max_value=3, value=1)



if st.button("Predict_CHD"):
    input_data = pd.DataFrame ([{
        "age":age,
        "sex":sex,
        "cp":cp,
        "trestbps":trestbps,
        "chol":chol,
        "fbs":fbs,
        "restecg":restecg,
        "thalach":thalach,
        "exang":exang,
        "oldpeak":oldpeak,
        "slope" : slope,
        "ca":ca,
        "thal":thal,
       
    }])
    
    input_data = input_data[list(imputer.feature_names_in_)]

    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    
    probability = model.predict_proba(input_data)[0][1]
    threshold = 0.2
    prediction = 1 if probability >= threshold else 0

    st.subheader("prediction_Result")

    if prediction ==1:
        st.error(f"CHD_DETECTE, PRABABILITY : {probability:.2f}")

    else:
        st.success(f"No_CHD_Detected, probability : {probability: 2f}")


# run with python -m streamlit run streamlit_code.py