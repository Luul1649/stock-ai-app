import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from newsapi import NewsApiClient
from textblob import TextBlob

# -------------------------------
# PAGE TITLE
# -------------------------------

st.title("AI Stock Price Prediction System")
st.subheader("LSTM + Global News Sentiment Analysis")

# -------------------------------
# USER INPUT
# -------------------------------

stock = st.text_input("Enter Stock Symbol", "AAPL")

# -------------------------------
# LOAD MODEL
# -------------------------------

model = load_model("lstm_model_cleaned.h5")

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# DOWNLOAD STOCK DATA
# -------------------------------

data = yf.download(stock,start="2015-01-01")

st.subheader("Recent Stock Data")
st.write(data.tail())

# -------------------------------
# PRICE CHART
# -------------------------------

st.subheader("Closing Price Chart")

fig = plt.figure(figsize=(10,5))
plt.plot(data['Close'])
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(stock)

st.pyplot(fig)

# -------------------------------
# DATA PREPARATION
# -------------------------------

close_prices = data['Close'].values.reshape(-1,1)

scaled_data = scaler.transform(close_prices)

sequence_length = 60

X = []

for i in range(sequence_length,len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i,0])

X = np.array(X)

X = np.reshape(X,(X.shape[0],X.shape[1],1))

# -------------------------------
# MODEL PREDICTION
# -------------------------------

predicted_prices = model.predict(X)

predicted_prices = scaler.inverse_transform(predicted_prices)

train = data[:sequence_length]

valid = data[sequence_length:].copy()

valid["Predictions"] = predicted_prices

# -------------------------------
# ACTUAL VS PREDICTED GRAPH
# -------------------------------

st.subheader("Actual vs Predicted Prices")

fig2 = plt.figure(figsize=(10,5))

plt.plot(train["Close"])
plt.plot(valid[["Close","Predictions"]])

plt.legend(["Train","Actual","Predicted"])

st.pyplot(fig2)

# -------------------------------
# NEXT DAY PREDICTION
# -------------------------------

last_60 = scaled_data[-60:]

last_60 = np.reshape(last_60,(1,60,1))

next_price = model.predict(last_60)

next_price = scaler.inverse_transform(next_price)

st.subheader("Predicted Next Day Closing Price")

st.success(f"${next_price[0][0]:.2f}")

# -------------------------------
# GLOBAL CURRENT AFFAIRS NEWS
# -------------------------------

st.subheader("Global News Affecting Markets")

NEWS_API_KEY = "428e0e12db8c48ffbb72b6efa59d632f"

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

articles = newsapi.get_everything(
    q=f"{stock} OR politics OR economy OR global markets OR inflation OR technology",
    language="en",
    sort_by="publishedAt",
    page_size=10
)

# -------------------------------
# SENTIMENT ANALYSIS FUNCTION
# -------------------------------

def get_sentiment(text):

    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive 📈"
    
    elif polarity < 0:
        return "Negative 📉"
    
    else:
        return "Neutral"

# -------------------------------
# DISPLAY NEWS
# -------------------------------

for article in articles["articles"]:

    title = article["title"]

    description = article["description"]

    url = article["url"]

    source = article["source"]["name"]

    sentiment = get_sentiment(title)

    st.markdown(f"### {title}")

    if description:
        st.write(description)

    st.write("Source:", source)

    st.write("Sentiment:", sentiment)

    st.markdown(f"[Read Full Article]({url})")

    st.write("---")
