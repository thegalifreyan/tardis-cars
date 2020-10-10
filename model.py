import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

#remove scientific notation
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import re #regular expression


#preprocessing
train = pd.read_csv('train-data.csv')
train = train.drop_duplicates()
train = train.drop(['Unnamed: 0', 'Name', 'New_Price'], axis=1)
train["Mileage"] = train["Mileage"].str.extract("(\d*\.?\d+)", expand=True)
train["Engine"] = train["Engine"].str.extract("(\d*\.?\d+)", expand=True)
train["Power"] = train["Power"].str.extract("(\d*\.?\d+)", expand=True)
train['Engine'] = pd.to_numeric(train['Engine'], downcast="float")
train['Power'] = pd.to_numeric(train['Power'], downcast="float")
train['Mileage'] = pd.to_numeric(train['Mileage'], downcast="float")

train[train.select_dtypes(['object']).columns] = train.select_dtypes(['object']).apply(lambda x: x.astype('category'))

train = train.replace({"Owner_Type": {"First": 1, "Second": 2, "Third": 3, "Fourth & Above": 4}})

for column in train.columns:
    train[column].fillna(train[column].mode()[0], inplace=True)

#Separating target and source columns
X = train.drop('Price', axis=1)
y = train['Price']

le = LabelEncoder()
le.fit(X['Location'])

X['Location'] = le.transform(X['Location'])
np.save('classes.npy', le.classes_)

X = pd.get_dummies(X)

#model building

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree= 0.7837281631746248,
 learning_rate= 0.09459955666386746,
 max_depth= 8,
 min_child_weight= 3,
 n_estimators= 502,
 subsample= 0.6435)
#xg_model = Pipeline(steps=[('scaler', StandardScaler(with_mean=False)), ('model', xg_reg)])
xg_reg.fit(X, y)

filename = 'used_cars_model.sav'
pickle.dump(xg_reg, open(filename, 'wb'))
