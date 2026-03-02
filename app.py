import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
from keras.models import load_model 
import streamlit as st
model = load_model(r"C:\Users\hi\Desktop\STOCK\Stock Predictions Model.keras")

st.header('STOCK MARKET PREDICTOR') 
stock=st.text_input('Enter Stock Symbol','GOOG')
start='2012-01-01'
end='2026-01-01'


data=yf.download(stock,start,end) 
st.subheader('Raw Data')
st.write(data)
data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)]) 

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
pas_100_days=scaler.fit_transform(data_train)
data_test_scaler=pd.concat([pas_100_days,data_test],ignore_index=True)
data_test_scaler=scaler.fit_transform(data_test)

x=[]
y=[]
for i in range(100,data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])
x,y=np.array(x),np.array(y)

