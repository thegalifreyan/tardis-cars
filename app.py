import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

np.load.__defaults__=(None, True, True, 'ASCII')

st.write("""
# Predicting Used Car values
This app takes values from the user and predicts what a used car would get you
 """)

st.sidebar.header('User Input Parameters')

def user_input_features():
    location = st.sidebar.selectbox(
        'Location',
        options=['Mumbai', 'Hyderabad', 'Kochi', 'Coimbatore', 'Pune', 'Delhi', 'Kolkata', 'Chennai', 'Jaipur', 'Bangalore', 'Ahmedabad'])
    year = st.sidebar.slider('Year', 1998, 2019, 2005)
    kmd = st.sidebar.slider('Kilometers Driven', 0, 9000000, 150000)
    ot = st.sidebar.radio('Owner Type', ('First', 'Second', 'Third', 'Fourth and Above'))
    mileage = st.sidebar.slider('Mileage in kmpl', 0, 40, 18)
    engine = st.sidebar.slider('Engine in CC', 0, 6000, 1500)
    power = st.sidebar.slider('Power in bhp', 0, 800, 120)
    seats = st.sidebar.slider('Seats', 0, 12, 5)
    fuel = st.sidebar.selectbox(
        'Fuel Type',
        options=['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol'])
    trans = st.sidebar.radio('Owner Type', ('Automatic', 'Manual'))

    otn = 0
    if ot=='First':
        otn = 1
    elif ot=='Second':
        otn = 2
    elif ot == 'Third':
        otn = 3
    elif ot == 'Fourth':
        otn = 4

    ftcng = 0
    ftdiesel = 0
    ftelectric = 0
    ftlpg = 0
    ftpatrol = 0
    transauto = 0
    transmanual = 0

    if fuel=='CNG':
        ftcng = 1
    elif fuel =='Diesel':
        ftdiesel = 1
    elif fuel == 'Electric':
        ftelectric = 1
    elif fuel == 'LPG':
        ftlpg = 1
    elif fuel == 'Petrol':
        ftpatrol = 1

    if trans == 'Automatic':
        transauto = 1
    elif trans == 'Manual':
        transmanual = 1

    data = {'Location': location,
            'Year': year,
            'Kilometers_Driven': kmd,
            'Owner_Type': otn,
            'Mileage': mileage,
            'Engine': engine,
            'Power': power,
            'Seats': seats,
            'Fuel_Type_CNG': ftcng,
            'Fuel_Type_Diesel': ftdiesel,
            'Fuel_Type_Electric': ftelectric,
            'Fuel_Type_LPG':ftlpg,
            'Fuel_Type_Petrol': ftpatrol,
            'Transmission_Automatic': transauto,
            'Transmission_Manual': transmanual }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

le = LabelEncoder()
le.classes_ = np.load('classes.npy')

df['Location'] = le.transform(df['Location'])

st.subheader('User Input parameters')
st.write(df)

#df1 = xg_model.transform(df)
filename = 'used_cars_model.sav'
xg_reg = pickle.load(open(filename, 'rb'))

pred = xg_reg.predict(df)

st.write("""
## Prediction of Car Value:
""")
st.write("\n{:0.2f} lakhs\n".format(pred[0]))

# restore np.load for future normal usage
np.load.__defaults__=(None, False, True, 'ASCII')
