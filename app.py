import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import requests
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from newsapi import NewsApiClient

# ------------------------------
# Page Title
# ------------------------------
st.title("AI Stock Price Predictor with LSTM + Financial News")

# ------------------------------
# User Input
# ------------------------------
stock = st.text_input("Enter Stock Symbol", "AAPL")

# ------------------------------
# Load Model
# ------------------------------
model = load_model("lstm_model_cleaned.h5")

# ------------------------------
# Load Scaler
# ------------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------------------
# Download Stock Data
# ------------------------------
data = yf.download(stock, start="2015-01-01")

st.subheader("Recent Stock Data")
st.write(data.tail())

# ------------------------------
# Plot Closing Price
# ------------------------------
st.subheader("Closing Price Chart")

fig = plt.figure(figsize=(10,5))
plt.plot(data['Close'])
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title(stock)

st.pyplot(fig)

# ------------------------------
# Prepare Data for LSTM
# ------------------------------
close_prices = data['Close'].values.reshape(-1,1)

scaled_data = scaler.transform(close_prices)

sequence_length = 60

X = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i,0])

X = np.array(X)
X = np.reshape(X,(X.shape[0],X.shape[1],1))

# ------------------------------
# Predictions
# ------------------------------
predicted_prices = model.predict(X)

predicted_prices = scaler.inverse_transform(predicted_prices)

train = data[:sequence_length]
valid = data[sequence_length:].copy()
valid['Predictions'] = predicted_prices

# ------------------------------
# Plot Predictions
# ------------------------------
st.subheader("Actual vs Predicted Prices")

fig2 = plt.figure(figsize=(10,5))

plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

plt.legend(['Train','Actual','Predicted'])

st.pyplot(fig2)

# ------------------------------
# Next Day Prediction
# ------------------------------
last_60 = scaled_data[-60:]
last_60 = np.reshape(last_60,(1,60,1))

next_price = model.predict(last_60)
next_price = scaler.inverse_transform(next_price)

st.subheader("Predicted Next Day Price")

st.success(f"${float(next_price):.2f}")

# ------------------------------
# Financial News Section
# ------------------------------

st.subheader("Latest Financial News")

NEWS_API_KEY = "428e0e12db8c48ffbb72b6efa59d632f"

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

articles = newsapi.get_everything(
    q=stock,
    language='en',
    sort_by='publishedAt',
    page_size=5
)

for article in articles['articles']:
    
    st.markdown(f"### {article['title']}")
    
    st.write(article['description'])
    
    st.write("Source:", article['source']['name'])
    
    st.markdown(f"[Read More]({article['url']})")
    
    st.write("---")

