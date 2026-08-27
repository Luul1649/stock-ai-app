import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from newsapi import NewsApiClient
from textblob import TextBlob
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("AI Stock Price Prediction System")
st.write("LSTM Deep Learning + Global News Sentiment")

# -----------------------------------
# SIDEBAR CONTROLS
# -----------------------------------
refresh = st.sidebar.slider("Auto Refresh (seconds)", 30, 300, 120)
stock = st.sidebar.text_input("Enter Stock Symbol", "AAPL").strip().upper()

# -----------------------------------
# CACHED FUNCTIONS (Prevents Log Spam & Rate Limits)
# -----------------------------------

@st.cache_resource
def load_ai_assets():
    """Loads the heavy LSTM model and scaler once and keeps them in memory."""
    model = load_model("lstm_model_cleaned.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data(ttl=600)  # Cache data for 10 minutes (600 seconds) to completely avoid Yahoo rate limits
def fetch_stock_data(ticker):
    """Downloads stock data and flattens multi-index columns cleanly."""
    df = yf.download(ticker, start="2015-01-01")
    if not df.empty and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

# -----------------------------------
# NATIVE APP FRAGMENT
# -----------------------------------
@st.fragment(run_every=refresh)
def run_app():
    # Load assets securely from memory cache
    try:
        model, scaler = load_ai_assets()
    except Exception as e:
        st.error(f"Error loading model or scaler files: {e}")
        return

    # Fetch data cleanly from memory/cache
    data = fetch_stock_data(stock)
    
    if data.empty:
        st.error("⚠️ Yahoo Finance Rate Limit Active or Invalid Ticker. Please wait a few minutes for the rate limit to reset, or try another ticker.")
        return

    st.subheader("Recent Stock Data")
    st.dataframe(data.tail())

    # -----------------------------------
    # PRICE CHART
    # -----------------------------------
    st.subheader("Stock Closing Price")
    fig = plt.figure(figsize=(10, 5))
    plt.plot(data["Close"])
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------
    # MOVING AVERAGES
    # -----------------------------------
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    st.subheader("Moving Averages")
    fig_ma = plt.figure(figsize=(10, 5))
    plt.plot(data["Close"], label="Close")
    plt.plot(data["MA50"], label="MA50")
    plt.plot(data["MA200"], label="MA200")
    plt.legend()
    st.pyplot(fig_ma)
    plt.close(fig_ma)

    # -----------------------------------
    # RSI FUNCTION & PLOT
    # -----------------------------------
    def compute_RSI(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data["RSI"] = compute_RSI(data["Close"])

    st.subheader("Relative Strength Index")
    fig_rsi = plt.figure(figsize=(10, 4))
    plt.plot(data["RSI"])
    plt.axhline(70, color='r', linestyle='--')
    plt.axhline(30, color='g', linestyle='--')
    st.pyplot(fig_rsi)
    plt.close(fig_rsi)

    # -----------------------------------
    # VOLATILITY
    # -----------------------------------
    data["Volatility"] = data["Close"].pct_change().rolling(20).std()

    st.subheader("Market Volatility")
    fig_vol = plt.figure(figsize=(10, 4))
    plt.plot(data["Volatility"])
    st.pyplot(fig_vol)
    plt.close(fig_vol)

    # -----------------------------------
    # DATA PREPARATION FOR LSTM
    # -----------------------------------
    close_prices = data["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)

    sequence_length = 60
    if len(scaled_data) <= sequence_length:
        st.warning("Not enough historical data rows to generate LSTM rolling sequences.")
        return

    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # -----------------------------------
    # PREDICTIONS
    # -----------------------------------
    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    train = data[:sequence_length]
    valid = data[sequence_length:].copy()
    valid["Predictions"] = predicted_prices

    # -----------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------
    st.subheader("Actual vs Predicted Prices")
    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(train["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Train", "Actual", "Predicted"])
    st.pyplot(fig2)
    plt.close(fig2)

    # -----------------------------------
    # NEXT DAY PREDICTION
    # -----------------------------------
    last_60 = scaled_data[-60:]
    last_60 = np.reshape(last_60, (1, 60, 1))

    next_price = model.predict(last_60)
    next_price = scaler.inverse_transform(next_price)

    st.subheader("Predicted Next Day Price")
    st.success(f"${next_price[0][0]:.2f}")

    # -----------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------
    rmse = math.sqrt(mean_squared_error(valid["Close"], valid["Predictions"]))
    mae = mean_absolute_error(valid["Close"], valid["Predictions"])

    st.subheader("Model Performance")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAE:** {mae:.4f}")

    # -----------------------------------
    # NEWS API
    # -----------------------------------
    st.subheader("Global News Affecting Markets")

    if "NEWS_API_KEY" in st.secrets:
        NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    else:
        NEWS_API_KEY = "428e0e12db8c48ffbb72b6efa59d632f"

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(
            q=f"{stock} OR politics OR economy OR global markets OR inflation OR technology",
            language="en",
            sort_by="publishedAt",
            page_size=8
        )

        def get_sentiment(text):
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0:
                return "Positive 📈"
            elif polarity < 0:
                return "Negative 📉"
            else:
                return "Neutral"

        for article in articles["articles"]:
            title = article["title"]
            description = article["description"]
            url = article["url"]
            source = article["source"]["name"]
            sentiment = get_sentiment(title)

            st.markdown(f"### {title}")
            if description:
                st.write(description)
            st.write(f"**Source:** {source} | **Sentiment:** {sentiment}")
            st.markdown(f"[Read Full Article]({url})")
            st.write("---")
            
    except Exception as news_err:
        st.error(f"Could not load news articles: {news_err}")

# Execute the integrated fragment app
run_app()
