import streamlit as st
import yfinance as yf
import requests
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import datetime

st.set_page_config(page_title="Real-Time Stock & Political Sentiment", layout="wide")
st.title(" Real-Time Stock Price & Political Sentiment Analyzer")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("User Inputs")

ticker = st.sidebar.text_input("Stock Symbol", "GOOG")
keyword = st.sidebar.text_input("Political Keyword", "politics")
country = st.sidebar.text_input("Country/Region", "USA")
interval = st.sidebar.selectbox("Stock Interval", ["1d","1h","5m"], index=0)
period = st.sidebar.selectbox("Data Period", ["1mo","3mo","6mo","1y"], index=0)
min_price = st.sidebar.number_input("Min Close Price", value=0)
max_price = st.sidebar.number_input("Max Close Price", value=10000)
min_volume = st.sidebar.number_input("Min Volume", value=0)
refresh = st.sidebar.checkbox("Auto Refresh Every 60 Seconds", value=False)

# -------------------------------
# Real-Time Refresh
# -------------------------------
if refresh:
    st_autorefresh(interval=60000, key="refresh")  # refresh every 60 seconds

# -------------------------------
# Step 1: Fetch Real-Time Stock Data
# -------------------------------
def fetch_stock(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error(f"No stock data found for {ticker}")
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# -------------------------------
# Step 2: Fetch Real-Time Political News & Sentiment
# -------------------------------
def fetch_political_news(keyword, country):
    url = "https://newsapi.org/v2/everything"
    today = datetime.date.today()
    params = {
        "q": f"{keyword} AND {country}",
        "from": str(today),
        "sortBy": "publishedAt",
        "apiKey": "YOUR_NEWSAPI_KEY"  # <-- Replace with your key
    }
    try:
        response = requests.get(url, params=params).json()
        headlines = [article['title'] for article in response.get('articles', [])]
        return headlines
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

def compute_sentiment(headlines):
    if not headlines:
        return 0
    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
    avg_sentiment = sum(sentiments)/len(sentiments)
    return avg_sentiment

# -------------------------------
# Main App Logic
# -------------------------------
if st.button("Analyze Real-Time Data"):
    # Stock Data
    stock_data = fetch_stock(ticker, period, interval)
    if stock_data.empty:
        st.stop()

    # Apply Filters
    filtered_stock = stock_data[
        (stock_data['Close'] >= min_price) & 
        (stock_data['Close'] <= max_price) &
        (stock_data['Volume'] >= min_volume)
    ]
    
    st.subheader(f"Filtered Stock Data for {ticker}")
    st.dataframe(filtered_stock.tail(10))
    
    # Political News + Sentiment
    headlines = fetch_political_news(keyword, country)
    st.subheader(f"Latest Political Headlines ({len(headlines)} found)")
    st.write(headlines[:10])
    
    avg_sentiment = compute_sentiment(headlines)
    if avg_sentiment > 0.05:
        st.success(f"Positive Political Sentiment → Market Optimism ({avg_sentiment:.2f})")
    elif avg_sentiment < -0.05:
        st.error(f"Negative Political Sentiment → Possible Market Drop ({avg_sentiment:.2f})")
    else:
        st.info(f"Neutral Political Atmosphere → Minor Effect ({avg_sentiment:.2f})")
    
    # Visualization
    filtered_stock['Sentiment'] = avg_sentiment  # same sentiment across dates
    fig, ax1 = plt.subplots(figsize=(10,5))
    
    ax1.plot(filtered_stock.index, filtered_stock['Close'], color='blue', label='Stock Price')
    ax1.set_xlabel("Date/Time")
    ax1.set_ylabel("Stock Price")
    
    ax2 = ax1.twinx()
    ax2.plot(filtered_stock.index, filtered_stock['Sentiment'], color='red', label='Political Sentiment')
    ax2.set_ylabel("Sentiment Score")
    
    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    st.pyplot(fig)

