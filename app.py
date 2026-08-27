import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from newsapi import NewsApiClient
from textblob import TextBlob
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Price Prediction System")
st.write("LSTM Deep Learning + Global News Sentiment")


# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.header("Stock Settings")

refresh = st.sidebar.slider(
    "Auto Refresh (seconds)",
    min_value=30,
    max_value=300,
    value=120
)

stock_input = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL"
)

stock = stock_input.strip().upper()

st.sidebar.info(
    "Examples: AAPL, TSLA, NVDA, MSFT, AMZN, GOOGL"
)


# ==========================================
# LOAD AI MODEL AND SCALER
# ==========================================

@st.cache_resource
def load_ai_assets():
    """
    Loads the trained LSTM model and scaler.
    These are cached so that the model is not
    loaded repeatedly every time the application refreshes.
    """

    model = load_model("lstm_model_cleaned.h5")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ==========================================
# FETCH STOCK DATA FROM YAHOO FINANCE
# ==========================================

@st.cache_data(ttl=300)
def fetch_global_stock_data(ticker):
    """
    Fetch historical stock data from Yahoo Finance Chart API.

    Returns:
        DataFrame containing:
        Open
        High
        Low
        Close
        Volume
    """

    ticker = ticker.strip().upper()

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    params = {
        "range": "10y",
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true"
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20
        )

        # Check HTTP response
        response.raise_for_status()

        # Convert response to JSON
        json_data = response.json()

        # Extract chart information
        chart = json_data.get("chart", {})

        result = chart.get("result")

        if not result:
            return pd.DataFrame()

        root = result[0]

        # Get timestamps
        timestamps = root.get("timestamp")

        if not timestamps:
            return pd.DataFrame()

        # Get quote information
        quote_list = (
            root
            .get("indicators", {})
            .get("quote", [])
        )

        if not quote_list:
            return pd.DataFrame()

        quote = quote_list[0]

        # Create DataFrame
        df = pd.DataFrame({
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume")
        })

        # Convert timestamps to dates
        df["Date"] = pd.to_datetime(
            timestamps,
            unit="s",
            utc=True
        ).dt.tz_localize(None)

        # Set Date as index
        df.set_index("Date", inplace=True)

        # Remove rows where Close is missing
        df.dropna(
            subset=["Close"],
            inplace=True
        )

        # Sort chronologically
        df.sort_index(inplace=True)

        return df

    except requests.exceptions.RequestException as e:

        st.error(
            f"Yahoo Finance connection error: {e}"
        )

        return pd.DataFrame()

    except ValueError as e:

        st.error(
            f"Yahoo Finance returned invalid data: {e}"
        )

        return pd.DataFrame()

    except Exception as e:

        st.error(
            f"Unexpected data error: {e}"
        )

        return pd.DataFrame()


# ==========================================
# RSI FUNCTION
# ==========================================

def compute_RSI(series, window=14):

    delta = series.diff()

    gain = (
        delta
        .where(delta > 0, 0)
        .rolling(window)
        .mean()
    )

    loss = (
        -delta
        .where(delta < 0, 0)
        .rolling(window)
        .mean()
    )

    rs = gain / loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


# ==========================================
# MAIN APPLICATION
# ==========================================

@st.fragment(run_every=refresh)
def run_app():

    # ======================================
    # LOAD MODEL
    # ======================================

    try:

        model, scaler = load_ai_assets()

    except Exception as e:

        st.error(
            "❌ Error loading the AI model or scaler."
        )

        st.error(str(e))

        return


    # ======================================
    # FETCH STOCK DATA
    # ======================================

    data = fetch_global_stock_data(stock)


    # ======================================
    # CHECK DATA
    # ======================================

    if data.empty:

        st.error(
            f"""
            ⚠️ Could not retrieve stock data for **{stock}**.

            Please check that the stock symbol is valid.

            Examples:
            **AAPL**, **TSLA**, **NVDA**, **MSFT**, **AMZN**
            """
        )

        return


    if len(data) < 65:

        st.error(
            f"""
            ⚠️ Not enough historical data for **{stock}**.

            Records retrieved: **{len(data)}**

            The LSTM model requires at least 65 historical
            records to generate predictions.
            """
        )

        return


    # ======================================
    # STOCK INFORMATION
    # ======================================

    st.subheader(
        f"📊 {stock} Stock Information"
    )

    latest_price = data["Close"].iloc[-1]

    previous_price = data["Close"].iloc[-2]

    price_change = (
        latest_price - previous_price
    )

    percentage_change = (
        price_change / previous_price
    ) * 100


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Latest Price",
            f"${latest_price:.2f}"
        )

    with col2:

        st.metric(
            "Daily Change",
            f"${price_change:.2f}"
        )

    with col3:

        st.metric(
            "Daily Change %",
            f"{percentage_change:.2f}%"
        )


    # ======================================
    # RECENT STOCK DATA
    # ======================================

    st.subheader("Recent Stock Data")

    st.dataframe(
        data.tail(10),
        use_container_width=True
    )


    # ======================================
    # STOCK CLOSING PRICE
    # ======================================

    st.subheader("📈 Stock Closing Price")

    fig = plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        data.index,
        data["Close"]
    )

    plt.xlabel("Date")

    plt.ylabel("Price ($)")

    plt.title(
        f"{stock} Closing Price"
    )

    plt.grid(True)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ======================================
    # MOVING AVERAGES
    # ======================================

    data["MA50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    data["MA200"] = (
        data["Close"]
        .rolling(200)
        .mean()
    )


    st.subheader(
        "📊 Moving Averages"
    )

    fig_ma = plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        data.index,
        data["Close"],
        label="Close"
    )

    plt.plot(
        data.index,
        data["MA50"],
        label="MA50"
    )

    plt.plot(
        data.index,
        data["MA200"],
        label="MA200"
    )

    plt.xlabel("Date")

    plt.ylabel("Price ($)")

    plt.title(
        f"{stock} Moving Averages"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    st.pyplot(fig_ma)

    plt.close(fig_ma)


    # ======================================
    # RSI
    # ======================================

    data["RSI"] = compute_RSI(
        data["Close"]
    )


    st.subheader(
        "📉 Relative Strength Index (RSI)"
    )

    fig_rsi = plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        data.index,
        data["RSI"],
        label="RSI"
    )

    plt.axhline(
        70,
        linestyle="--"
    )

    plt.axhline(
        30,
        linestyle="--"
    )

    plt.xlabel("Date")

    plt.ylabel("RSI")

    plt.title(
        f"{stock} RSI"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    st.pyplot(fig_rsi)

    plt.close(fig_rsi)


    # ======================================
    # VOLATILITY
    # ======================================

    data["Volatility"] = (
        data["Close"]
        .pct_change()
        .rolling(20)
        .std()
    )


    st.subheader(
        "📊 Market Volatility"
    )

    fig_vol = plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        data.index,
        data["Volatility"]
    )

    plt.xlabel("Date")

    plt.ylabel("Volatility")

    plt.title(
        f"{stock} 20-Day Rolling Volatility"
    )

    plt.grid(True)

    plt.tight_layout()

    st.pyplot(fig_vol)

    plt.close(fig_vol)


    # ======================================
    # PREPARE DATA FOR LSTM
    # ======================================

    close_prices = (
        data["Close"]
        .values
        .reshape(-1, 1)
    )


    try:

        scaled_data = scaler.transform(
            close_prices
        )

    except Exception as e:

        st.error(
            "❌ Error scaling stock data."
        )

        st.error(str(e))

        return


    sequence_length = 60


    X = []


    for i in range(
        sequence_length,
        len(scaled_data)
    ):

        X.append(
            scaled_data[
                i - sequence_length:i,
                0
            ]
        )


    X = np.array(X)


    X = np.reshape(
        X,
        (
            X.shape[0],
            X.shape[1],
            1
        )
    )


    # ======================================
    # MODEL PREDICTIONS
    # ======================================

    try:

        predicted_prices = model.predict(
            X,
            verbose=0
        )

        predicted_prices = (
            scaler.inverse_transform(
                predicted_prices
            )
        )

    except Exception as e:

        st.error(
            "❌ Error generating LSTM predictions."
        )

        st.error(str(e))

        return


    # ======================================
    # ACTUAL VS PREDICTED
    # ======================================

    train = data.iloc[
        :sequence_length
    ]

    valid = data.iloc[
        sequence_length:
    ].copy()


    valid["Predictions"] = (
        predicted_prices.flatten()
    )


    st.subheader(
        "🤖 Actual vs Predicted Prices"
    )


    fig2 = plt.figure(
        figsize=(12, 5)
    )


    plt.plot(
        train.index,
        train["Close"],
        label="Training Data"
    )


    plt.plot(
        valid.index,
        valid["Close"],
        label="Actual Price"
    )


    plt.plot(
        valid.index,
        valid["Predictions"],
        label="Predicted Price"
    )


    plt.xlabel("Date")

    plt.ylabel("Price ($)")

    plt.title(
        f"{stock} Actual vs LSTM Predicted Prices"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)


    # ======================================
    # NEXT DAY PREDICTION
    # ======================================

    last_60 = scaled_data[-60:]


    last_60 = np.reshape(
        last_60,
        (1, 60, 1)
    )


    try:

        next_price_scaled = model.predict(
            last_60,
            verbose=0
        )


        next_price = (
            scaler.inverse_transform(
                next_price_scaled
            )
        )


        next_day_price = (
            float(next_price[0][0])
        )

    except Exception as e:

        st.error(
            "❌ Could not generate next-day prediction."
        )

        st.error(str(e))

        return


    # ======================================
    # DISPLAY NEXT DAY PREDICTION
    # ======================================

    st.subheader(
        "🔮 Predicted Next Day Price"
    )


    prediction_difference = (
        next_day_price - latest_price
    )


    prediction_percentage = (
        prediction_difference /
        latest_price
    ) * 100


    col1, col2, col3 = st.columns(3)


    with col1:

        st.success(
            f"${next_day_price:.2f}"
        )


    with col2:

        st.metric(
            "Predicted Change",
            f"${prediction_difference:.2f}"
        )


    with col3:

        st.metric(
            "Predicted Change %",
            f"{prediction_percentage:.2f}%"
        )


    # ======================================
    # MODEL PERFORMANCE
    # ======================================

    rmse = math.sqrt(
        mean_squared_error(
            valid["Close"],
            valid["Predictions"]
        )
    )


    mae = mean_absolute_error(
        valid["Close"],
        valid["Predictions"]
    )


    st.subheader(
        "📊 Model Performance"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "RMSE",
            f"{rmse:.4f}"
        )


    with col2:

        st.metric(
            "MAE",
            f"{mae:.4f}"
        )


    # ======================================
    # DOWNLOAD PREDICTIONS
    # ======================================

    prediction_download = valid[
        ["Close", "Predictions"]
    ].copy()


    prediction_download[
        "Difference"
    ] = (
        prediction_download["Predictions"]
        - prediction_download["Close"]
    )


    csv_data = (
        prediction_download
        .to_csv()
        .encode("utf-8")
    )


    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_data,
        file_name=f"{stock}_predictions.csv",
        mime="text/csv"
    )


    # ======================================
    # NEWS SENTIMENT
    # ======================================

    st.subheader(
        "🌍 Global News Affecting Markets"
    )


    # ======================================
    # NEWS API KEY
    # ======================================

    if "NEWS_API_KEY" in st.secrets:

        NEWS_API_KEY = (
            st.secrets["NEWS_API_KEY"]
        )

    else:

        NEWS_API_KEY = None


    if not NEWS_API_KEY:

        st.warning(
            "⚠️ News API key is not configured. "
            "Add NEWS_API_KEY to Streamlit Secrets "
            "to enable news sentiment analysis."
        )

    else:

        try:

            newsapi = NewsApiClient(
                api_key=NEWS_API_KEY
            )


            articles = newsapi.get_everything(

                q=(
                    f"{stock} OR politics OR "
                    "economy OR global markets OR "
                    "inflation OR technology"
                ),

                language="en",

                sort_by="publishedAt",

                page_size=8
            )


            # ==================================
            # SENTIMENT FUNCTION
            # ==================================

            def get_sentiment(text):

                if not text:

                    return "Neutral", 0.0


                analysis = TextBlob(
                    text
                )


                polarity = (
                    analysis.sentiment.polarity
                )


                if polarity > 0:

                    sentiment = (
                        "Positive 📈"
                    )

                elif polarity < 0:

                    sentiment = (
                        "Negative 📉"
                    )

                else:

                    sentiment = "Neutral"


                return sentiment, polarity


            # ==================================
            # DISPLAY NEWS
            # ==================================

            if not articles.get("articles"):

                st.info(
                    "No recent news articles were found."
                )

            else:

                for article in articles["articles"]:

                    title = (
                        article.get(
                            "title",
                            "No title"
                        )
                    )


                    description = (
                        article.get(
                            "description"
                        )
                    )


                    url = (
                        article.get(
                            "url"
                        )
                    )


                    source_data = (
                        article.get(
                            "source",
                            {}
                        )
                    )


                    source = (
                        source_data.get(
                            "name",
                            "Unknown"
                        )
                    )


                    sentiment, polarity = (
                        get_sentiment(title)
                    )


                    st.markdown(
                        f"### {title}"
                    )


                    if description:

                        st.write(
                            description
                        )


                    st.write(
                        f"**Source:** {source}"
                    )


                    st.write(
                        f"**Sentiment:** {sentiment}"
                    )


                    st.write(
                        f"**Sentiment Score:** "
                        f"{polarity:.2f}"
                    )


                    if url:

                        st.markdown(
                            f"[Read Full Article]({url})"
                        )


                    st.write("---")


        except Exception as news_err:

            st.error(
                f"Could not load news articles: "
                f"{news_err}"
            )


# ==========================================
# RUN APPLICATION
# ==========================================

run_app()
