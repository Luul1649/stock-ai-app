import sys
print("Python used by Streamlit:", sys.executable)
import multitasking
print("Multitasking imported successfully!")
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import yfinance as yf
from keras.models import load_model 
import streamlit as st
model = load_model("Stock Predictions Model.keras")

st.header('STOCK MARKET PREDICTOR') 
stock=st.text_input('Enter Stock Symbol','GOOG')
start='2012-01-01'
end='2026-01-01'


data=yf.download(stock,start,end) 
st.subheader('Stock Data')
st.write(data)
data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)]) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(data_train)
data_test_scaler = scaler.transform(data_test)
# Fit only on training data
scaler.fit(data_train)

# Transform test data
data_test_scaler = scaler.transform(data_test)

# Fit only on training data
scaler.fit(data_train)

# Transform test data
data_test_scaler = scaler.transform(data_test)
st.header('Price vs MA50')
ma_50_days=data.Close.rolling(50).mean()
fig1=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)
st.header('Price vs MA50 vs MA100')
ma_100_days=data.Close.rolling(100).mean()
fig2=plt.figure(figsize=(10,8))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2)
st.header('Price vs MA50 vs MA100 vs MA200')
ma_200_days=data.Close.rolling(200).mean()
fig3=plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig3)
x=[]
y=[]
for i in range(100,data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])
x,y=np.array(x),np.array(y)

predict=model.predict(x)
scale=1/scaler.scale_
predict=predict*scale
y=y*scale













