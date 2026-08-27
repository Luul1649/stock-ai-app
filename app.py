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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Stock Prediction System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Price Prediction System")
st.markdown(
    "### LSTM Deep Learning + Real-Time Market Data + News Sentiment"
)

st.markdown("---")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Stock Controls")

stock_input = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL",
    help="Examples: AAPL, TSLA, NVDA, MSFT, AMZN"
)

stock = stock_input.strip().upper()

refresh = st.sidebar.slider(
    "Auto Refresh (seconds)",
    min_value=30,
    max_value=300,
    value=120
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Supported examples**

    AAPL – Apple  
    TSLA – Tesla  
    NVDA – NVIDIA  
    MSFT – Microsoft  
    AMZN – Amazon
    """
)


# ============================================================
# LOAD LSTM MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_ai_assets():

    model = load_model(
        "lstm_model_cleaned.h5"
    )

    with open(
        "scaler.pkl",
        "rb"
    ) as file:

        scaler = pickle.load(file)

    return model, scaler


# ============================================================
# FETCH STOCK DATA
# ============================================================

@st.cache_data(ttl=300)
@st.cache_data(ttl=300)
def fetch_stock_data(ticker):

    try:
        import yfinance as yf

        ticker = ticker.strip().upper()

        stock = yf.Ticker(ticker)

        data = stock.history(
            period="10y",
            interval="1d",
            auto_adjust=False
        )

        if data.empty:
            return pd.DataFrame()

        data = data[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ]

        data.dropna(
            subset=["Close"],
            inplace=True
        )

        data.sort_index(
            inplace=True
        )

        return data

    except Exception as e:

        st.error(
            f"Yahoo Finance error: {e}"
        )

        return pd.DataFrame()

        # ----------------------------------------------------
        # CHECK API RESPONSE
        # ----------------------------------------------------

        if result.get("status") == "error":

            st.error(
                "Twelve Data Error: "
                + result.get(
                    "message",
                    "Unknown API error"
                )
            )

            return pd.DataFrame()


        if "values" not in result:

            st.error(
                f"No historical stock data "
                f"was returned for {ticker}."
            )

            return pd.DataFrame()


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        df = pd.DataFrame(
            result["values"]
        )


        # ----------------------------------------------------
        # CONVERT DATE COLUMN
        # ----------------------------------------------------

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )


        # Remove invalid dates

        df = df.dropna(
            subset=["datetime"]
        )


        # ----------------------------------------------------
        # SET DATE AS INDEX
        # ----------------------------------------------------

        df.set_index(
            "datetime",
            inplace=True
        )


        # ----------------------------------------------------
        # RENAME COLUMNS
        # ----------------------------------------------------

        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            },
            inplace=True
        )


        # ----------------------------------------------------
        # CONVERT VALUES TO NUMERIC
        # ----------------------------------------------------

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )


        # ----------------------------------------------------
        # REMOVE INVALID ROWS
        # ----------------------------------------------------

        df.dropna(
            subset=["Close"],
            inplace=True
        )


        # ----------------------------------------------------
        # SORT CHRONOLOGICALLY
        # ----------------------------------------------------

        df.sort_index(
            ascending=True,
            inplace=True
        )


        return df


    except KeyError:

        st.error(
            """
            ❌ Twelve Data API key not found.

            Add TWELVE_DATA_API_KEY to your
            Streamlit Secrets.
            """
        )

        return pd.DataFrame()


    except requests.exceptions.RequestException as error:

        st.error(
            f"❌ Stock data connection error: {error}"
        )

        return pd.DataFrame()


    except Exception as error:

        st.error(
            f"❌ Unexpected data error: {error}"
        )

        return pd.DataFrame()


# ============================================================
# RSI FUNCTION
# ============================================================

def calculate_rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )

    average_gain = gain.rolling(
        period
    ).mean()

    average_loss = loss.rolling(
        period
    ).mean()

    rs = (
        average_gain /
        average_loss
    )

    rsi = (
        100 -
        (100 / (1 + rs))
    )

    return rsi


# ============================================================
# MAIN APPLICATION
# ============================================================

@st.fragment(run_every=refresh)
def run_app():

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        model, scaler = load_ai_assets()

    except Exception as error:

        st.error(
            "❌ Unable to load the LSTM model or scaler."
        )

        st.code(
            str(error)
        )

        st.info(
            """
            Make sure these files are in the same
            folder as app.py:

            • lstm_model_cleaned.h5
            • scaler.pkl
            """
        )

        return


    # ========================================================
    # FETCH STOCK DATA
    # ========================================================

    data = fetch_stock_data(
        stock
    )


    # ========================================================
    # CHECK DATA
    # ========================================================

    if data.empty:

        st.error(
            f"""
            ⚠️ Could not retrieve stock data
            for **{stock}**.
            """
        )

        return


    if len(data) < 65:

        st.error(
            f"""
            ⚠️ Insufficient historical data.

            Stock: {stock}

            Records received: {len(data)}

            Minimum required: 65
            """
        )

        return


    # ========================================================
    # LATEST STOCK INFORMATION
    # ========================================================

    latest_price = float(
        data["Close"].iloc[-1]
    )

    previous_price = float(
        data["Close"].iloc[-2]
    )

    daily_change = (
        latest_price -
        previous_price
    )

    daily_change_percent = (
        daily_change /
        previous_price
    ) * 100


    # ========================================================
    # STOCK SUMMARY
    # ========================================================

    st.subheader(
        f"📊 {stock} Market Overview"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Current Price",
            f"${latest_price:.2f}"
        )


    with col2:

        st.metric(
            "Daily Change",
            f"${daily_change:.2f}"
        )


    with col3:

        st.metric(
            "Daily Change %",
            f"{daily_change_percent:.2f}%"
        )


    with col4:

        latest_volume = data[
            "Volume"
        ].iloc[-1]

        st.metric(
            "Volume",
            f"{latest_volume:,.0f}"
        )


    # ========================================================
    # RECENT DATA
    # ========================================================

    st.subheader(
        "📋 Recent Stock Data"
    )

    st.dataframe(
        data.tail(10),
        use_container_width=True
    )


    # ========================================================
    # CLOSING PRICE CHART
    # ========================================================

    st.subheader(
        "📈 Stock Closing Price"
    )


    fig = plt.figure(
        figsize=(12, 5)
    )


    plt.plot(
        data.index,
        data["Close"],
        label="Closing Price"
    )


    plt.title(
        f"{stock} Historical Closing Price"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Price ($)"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


    # ========================================================
    # MOVING AVERAGES
    # ========================================================

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
        label="Closing Price"
    )


    plt.plot(
        data.index,
        data["MA50"],
        label="50-Day MA"
    )


    plt.plot(
        data.index,
        data["MA200"],
        label="200-Day MA"
    )


    plt.title(
        f"{stock} Moving Average Analysis"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Price ($)"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig_ma
    )

    plt.close(
        fig_ma
    )


    # ========================================================
    # RSI
    # ========================================================

    data["RSI"] = calculate_rsi(
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
        linestyle="--",
        label="Overbought (70)"
    )


    plt.axhline(
        30,
        linestyle="--",
        label="Oversold (30)"
    )


    plt.title(
        f"{stock} RSI"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "RSI"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig_rsi
    )

    plt.close(
        fig_rsi
    )


    # ========================================================
    # VOLATILITY
    # ========================================================

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
        data["Volatility"],
        label="20-Day Volatility"
    )


    plt.title(
        f"{stock} Market Volatility"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Volatility"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig_vol
    )

    plt.close(
        fig_vol
    )


    # ========================================================
    # LSTM DATA PREPARATION
    # ========================================================

    close_prices = (
        data["Close"]
        .values
        .reshape(-1, 1)
    )


    try:

        scaled_data = scaler.transform(
            close_prices
        )

    except Exception as error:

        st.error(
            "❌ Error applying the trained scaler."
        )

        st.code(
            str(error)
        )

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


    X = np.array(
        X
    )


    X = np.reshape(
        X,
        (
            X.shape[0],
            X.shape[1],
            1
        )
    )


    # ========================================================
    # LSTM PREDICTION
    # ========================================================

    try:

        predicted_scaled = model.predict(
            X,
            verbose=0
        )


        predicted_prices = (
            scaler.inverse_transform(
                predicted_scaled
            )
        )


    except Exception as error:

        st.error(
            "❌ Error generating LSTM predictions."
        )

        st.code(
            str(error)
        )

        return


    # ========================================================
    # ACTUAL VS PREDICTED DATA
    # ========================================================

    train = data.iloc[
        :sequence_length
    ]


    valid = data.iloc[
        sequence_length:
    ].copy()


    valid["Predictions"] = (
        predicted_prices.flatten()
    )


    # ========================================================
    # ACTUAL VS PREDICTED CHART
    # ========================================================

    st.subheader(
        "🤖 Actual vs LSTM Predicted Prices"
    )


    fig_prediction = plt.figure(
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


    plt.title(
        f"{stock} Actual vs Predicted Prices"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Price ($)"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(
        fig_prediction
    )

    plt.close(
        fig_prediction
    )


    # ========================================================
    # NEXT DAY PREDICTION
    # ========================================================

    last_60_days = scaled_data[
        -sequence_length:
    ]


    last_60_days = np.reshape(
        last_60_days,
        (1, sequence_length, 1)
    )


    try:

        next_prediction_scaled = (
            model.predict(
                last_60_days,
                verbose=0
            )
        )


        next_prediction = (
            scaler.inverse_transform(
                next_prediction_scaled
            )
        )


        next_price = float(
            next_prediction[0][0]
        )


    except Exception as error:

        st.error(
            "❌ Could not generate next-day prediction."
        )

        st.code(
            str(error)
        )

        return


    # ========================================================
    # PREDICTED PRICE
    # ========================================================

    st.subheader(
        "🔮 Next-Day Stock Price Prediction"
    )


    expected_change = (
        next_price -
        latest_price
    )


    expected_change_percent = (
        expected_change /
        latest_price
    ) * 100


    col1, col2, col3 = st.columns(3)


    with col1:

        st.success(
            f"${next_price:.2f}"
        )


    with col2:

        st.metric(
            "Expected Change",
            f"${expected_change:.2f}"
        )


    with col3:

        st.metric(
            "Expected Change %",
            f"{expected_change_percent:.2f}%"
        )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

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


    # ========================================================
    # DOWNLOAD PREDICTIONS
    # ========================================================

    download_data = valid[
        [
            "Close",
            "Predictions"
        ]
    ].copy()


    download_data["Difference"] = (
        download_data["Predictions"] -
        download_data["Close"]
    )


    csv = download_data.to_csv(
        index=True
    ).encode(
        "utf-8"
    )


    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name=f"{stock}_prediction_results.csv",
        mime="text/csv"
    )


    # ========================================================
    # NEWS SENTIMENT ANALYSIS
    # ========================================================

    st.subheader(
        "🌍 Global News Affecting Markets"
    )


    # --------------------------------------------------------
    # NEWS API KEY
    # --------------------------------------------------------

    try:

        news_api_key = st.secrets[
            "NEWS_API_KEY"
        ]

    except KeyError:

        news_api_key = None


    if not news_api_key:

        st.warning(
            """
            ⚠️ News API key is not configured.

            Add NEWS_API_KEY to Streamlit Secrets
            to activate real-time news sentiment analysis.
            """
        )

    else:

        try:

            newsapi = NewsApiClient(
                api_key=news_api_key
            )


            articles = newsapi.get_everything(

                q=(
                    f"{stock} OR "
                    "stock market OR "
                    "economy OR "
                    "inflation OR "
                    "technology OR "
                    "global markets"
                ),

                language="en",

                sort_by="publishedAt",

                page_size=8
            )


            # ------------------------------------------------
            # SENTIMENT FUNCTION
            # ------------------------------------------------

            def analyze_sentiment(text):

                if not text:

                    return (
                        "Neutral",
                        0.0
                    )


                analysis = TextBlob(
                    text
                )


                polarity = (
                    analysis
                    .sentiment
                    .polarity
                )


                if polarity > 0.05:

                    sentiment = (
                        "Positive 📈"
                    )

                elif polarity < -0.05:

                    sentiment = (
                        "Negative 📉"
                    )

                else:

                    sentiment = (
                        "Neutral ➖"
                    )


                return (
                    sentiment,
                    polarity
                )


            # ------------------------------------------------
            # DISPLAY ARTICLES
            # ------------------------------------------------

            article_list = articles.get(
                "articles",
                []
            )


            if not article_list:

                st.info(
                    "No recent news articles were found."
                )


            else:

                for article in article_list:

                    title = article.get(
                        "title",
                        "No title available"
                    )


                    description = article.get(
                        "description"
                    )


                    url = article.get(
                        "url"
                    )


                    source_info = article.get(
                        "source",
                        {}
                    )


                    source = source_info.get(
                        "name",
                        "Unknown"
                    )


                    sentiment, score = (
                        analyze_sentiment(
                            title
                        )
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
                        f"{score:.2f}"
                    )


                    if url:

                        st.markdown(
                            f"[Read Full Article]({url})"
                        )


                    st.markdown("---")


        except Exception as error:

            st.error(
                f"❌ Could not load news articles: {error}"
            )


# ============================================================
# RUN APPLICATION
# ============================================================

run_app()
