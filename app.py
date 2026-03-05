import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
import datetime
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle

# -------------------------------
# 1️⃣ Streamlit App Settings
# -------------------------------
st.set_page_config(page_title="Stock Forecast + Political Sentiment", layout="wide")
st.title("📈 Stock Forecast (5 Days) with Political Sentiment")

# -------------------------------
# 2️⃣ Sidebar Inputs
# -------------------------------
st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Stock Symbol", "GOOG")
keyword = st.sidebar.text_input("Political Keyword", "politics")
country = st.sidebar.text_input("Country", "USA")
interval = st.sidebar.selectbox("Interval", ["1d","1h","5m"])
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y"])
min_price = st.sidebar.number_input("Min Close Price", 0)
max_price = st.sidebar.number_input("Max Close Price", 10000)
min_volume = st.sidebar.number_input("Min Volume", 0)

# -------------------------------
# 3️⃣ Load LSTM Model and Scaler
# -------------------------------
@st.cache_resource
def load_model_scaler():
    model = load_model("lstm_model_cleaned.h5")
    scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

model, scaler = load_model_scaler()
timesteps = 100
feature_col = "Close"

# -------------------------------
# 4️⃣ Fetch Stock Data
# -------------------------------
def fetch_stock(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        st.error(f"No stock data found for {ticker}")
    return data

stock_data = fetch_stock(ticker, period, interval)

# Apply filters
filtered_stock = stock_data[
    (stock_data['Close'] >= min_price) &
    (stock_data['Close'] <= max_price) &
    (stock_data['Volume'] >= min_volume)
]

st.subheader("Filtered Stock Data")
st.dataframe(filtered_stock.tail(10))

# -------------------------------
# 5️⃣ Fetch Political News & Sentiment
# -------------------------------
def fetch_news(keyword="politics", country="USA"):
    url = "https://newsapi.org/v2/everything"
    today = datetime.date.today()
    params = {
        "q": f"{keyword} AND {country}",
        "from": str(today),
        "sortBy": "publishedAt",
        "apiKey": "YOUR_NEWSAPI_KEY"  # Replace with your NewsAPI key
    }
    try:
        response = requests.get(url, params=params).json()
        headlines = [article['title'] for article in response.get('articles',[])]
        return headlines
    except:
        st.error("Error fetching news")
        return []

def compute_sentiment(headlines):
    if not headlines:
        return 0
    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
    return sum(sentiments)/len(sentiments)

headlines = fetch_news(keyword, country)
st.subheader(f"Political Headlines ({len(headlines)} found)")
st.write(headlines[:10])

avg_sentiment = compute_sentiment(headlines)
if avg_sentiment > 0.05:
    st.success(f"Positive Political Sentiment ({avg_sentiment:.2f})")
elif avg_sentiment < -0.05:
    st.error(f"Negative Political Sentiment ({avg_sentiment:.2f})")
else:
    st.info(f"Neutral Political Sentiment ({avg_sentiment:.2f})")

# -------------------------------
# 6️⃣ 5-Day Forecast
# -------------------------------
if len(filtered_stock) < timesteps:
    st.warning(f"Not enough data for prediction (need at least {timesteps} records)")
else:
    future_days = 5
    predicted_prices = []
    
    last_days = filtered_stock[feature_col].tail(timesteps).values
    input_seq = last_days.copy()
    
    for _ in range(future_days):
        X_input = input_seq.reshape(1, timesteps, 1)
        pred_scaled = model.predict(X_input)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        
        # Adjust by sentiment (example: +/- 0.5%)
        adjustment = pred_price * (avg_sentiment * 0.005)
        pred_price_adjusted = pred_price + adjustment
        
        predicted_prices.append(pred_price_adjusted)
        
        # Update input_seq with scaled adjusted prediction
        scaled_adjusted = scaler.transform(np.array([[pred_price_adjusted]]))
        input_seq = np.append(input_seq[1:], scaled_adjusted, axis=0)
    
    st.subheader(f"Predicted Close Prices for Next {future_days} Days:")
    for i, price in enumerate(predicted_prices, 1):
        st.write(f"Day {i}: ${price:.2f}")

# -------------------------------
# 7️⃣ Plot Forecast
# -------------------------------
last_date = filtered_stock.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

plt.figure(figsize=(10,5))
plt.plot(filtered_stock.index, filtered_stock['Close'], label="Actual Close", color="blue")
plt.plot(future_dates, predicted_prices, label="Predicted Close (Next 5 Days)", color="green", marker="o")
plt.axhline(y=filtered_stock['Close'].mean(), color='red', linestyle='--', label=f"Political Sentiment ({avg_sentiment:.2f})")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{ticker} 5-Day Forecast with Political Sentiment")
plt.legend()
st.pyplot(plt)

