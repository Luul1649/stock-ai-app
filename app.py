import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = load_model("Stock Predictions Model.keras")

st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2012-01-01'
end = '2026-01-01'

data = yf.download(stock, start ,end)
import yfinance as yf
import streamlit as st

ticker = st.text_input("Enter Stock Symbol", "GOOG")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if st.button("Fetch Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for this period!")
    else:
        st.dataframe(data)

st.subheader('Stock Data')
st.write(data)
min_price = st.number_input("Min Close Price", value=0)
max_price = st.number_input("Max Close Price", value=10000)

filtered_data = data[(data['Close'] >= min_price) & (data['Close'] <= max_price)]
st.dataframe(filtered_data)
data['SMA_20'] = data['Close'].rolling(window=20).mean()
sma_filter = st.slider("Show data where Close > SMA_20", 0, 1, 1)

if sma_filter:
    filtered_data = data[data['Close'] > data['SMA_20']]
    st.line_chart(filtered_data['Close'])

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)
min_volume = st.number_input("Minimum Volume", value=0)
filtered_data = data[data['Volume'] >= min_volume]
st.dataframe(filtered_data)
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)
y_true = y   # actual values
y_pred = predict  # model predictions
scale = 1/scaler.scale_

predict = predict * scale
y = y * scale
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)
mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)
r2 = r2_score(y_true, y_pred)
print("R2 Score:", r2)
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

st.pyplot(fig4)



