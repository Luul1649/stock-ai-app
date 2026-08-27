import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pickle
import math

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from newsapi import NewsApiClient
from textblob import TextBlob


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Stock Price Prediction System",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("AI Stock Price Prediction System")

st.subheader(
    "LSTM Deep Learning + Real-Time Market Data + News Sentiment"
)

st.markdown("---")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Stock Controls")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL"
).strip().upper()

refresh_time = st.sidebar.slider(
    "Auto Refresh (seconds)",
    min_value=60,
    max_value=300,
    value=120
)

st.sidebar.markdown("---")

st.sidebar.write("Examples:")

st.sidebar.write(
    "AAPL • TSLA • NVDA • MSFT • AMZN"
)


# ============================================================
# LOAD MODEL AND SCALER
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
def fetch_stock_data(ticker):

    try:

        ticker = ticker.strip().upper()

        # Use Yahoo Finance through yfinance
        data = yf.download(
            ticker,
            period="10y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        # ----------------------------------------------------
        # CHECK DATA
        # ----------------------------------------------------

        if data is None or data.empty:
            return pd.DataFrame()

        # ----------------------------------------------------
        # HANDLE MULTI-INDEX COLUMNS
        # ----------------------------------------------------

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        # Check missing columns

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:

            st.error(
                f"Missing Yahoo Finance columns: {missing}"
            )

            return pd.DataFrame()

        # ----------------------------------------------------
        # KEEP REQUIRED DATA
        # ----------------------------------------------------

        data = data[
            required_columns
        ].copy()

        # ----------------------------------------------------
        # CONVERT NUMERIC DATA
        # ----------------------------------------------------

        for column in required_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # REMOVE MISSING CLOSE PRICES
        # ----------------------------------------------------

        data.dropna(
            subset=["Close"],
            inplace=True
        )

        # ----------------------------------------------------
        # SORT BY DATE
        # ----------------------------------------------------

        data.sort_index(
            ascending=True,
            inplace=True
        )

        return data


    except Exception as error:

        st.error(
            f"Yahoo Finance error: {error}"
        )

        return pd.DataFrame()


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    prices,
    period=14
):

    difference = prices.diff()

    gain = difference.clip(
        lower=0
    )

    loss = -difference.clip(
        upper=0
    )

    average_gain = gain.rolling(
        period
    ).mean()

    average_loss = loss.rolling(
        period
    ).mean()

    rs = (
        average_gain /
        average_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    return rsi


# ============================================================
# NEWS SENTIMENT
# ============================================================

def analyze_sentiment(text):

    if not text:

        return (
            "Neutral",
            0.0
        )

    analysis = TextBlob(
        str(text)
    )

    polarity = (
        analysis.sentiment.polarity
    )

    if polarity > 0.05:

        sentiment = "Positive 📈"

    elif polarity < -0.05:

        sentiment = "Negative 📉"

    else:

        sentiment = "Neutral ➖"

    return (
        sentiment,
        polarity
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # ========================================================
    # LOAD AI MODEL
    # ========================================================

    try:

        model, scaler = load_ai_assets()

    except Exception as error:

        st.error(
            "Unable to load the AI model."
        )

        st.code(
            str(error)
        )

        st.info(
            """
            Make sure these files are in your
            GitHub repository:

            lstm_model_cleaned.h5
            scaler.pkl
            """
        )

        return


    # ========================================================
    # FETCH STOCK DATA
    # ========================================================

    with st.spinner(
        f"Loading {stock} market data..."
    ):

        data = fetch_stock_data(
            stock
        )


    # ========================================================
    # DATA VALIDATION
    # ========================================================

    if data.empty:

        st.error(
            f"""
            Could not retrieve stock data for **{stock}**.

            Please check that the stock symbol is correct.
            """
        )

        return


    if len(data) < 65:

        st.error(
            f"""
            Not enough historical data for {stock}.

            Data received: {len(data)} rows.

            At least 65 rows are required for
            the 60-day LSTM sequence.
            """
        )

        return


    # ========================================================
    # CURRENT PRICE
    # ========================================================

    current_price = float(
        data["Close"].iloc[-1]
    )

    previous_price = float(
        data["Close"].iloc[-2]
    )

    price_change = (
        current_price -
        previous_price
    )

    percentage_change = (
        price_change /
        previous_price
    ) * 100


    # ========================================================
    # MARKET SUMMARY
    # ========================================================

    st.header(
        f"📊 {stock} Market Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Price",
            f"${current_price:.2f}"
        )

    with col2:

        st.metric(
            "Daily Change",
            f"${price_change:.2f}"
        )

    with col3:

        st.metric(
            "Change %",
            f"{percentage_change:.2f}%"
        )

    with col4:

        latest_volume = int(
            data["Volume"].iloc[-1]
        )

        st.metric(
            "Volume",
            f"{latest_volume:,}"
        )


    # ========================================================
    # RECENT DATA
    # ========================================================

    st.header(
        "📋 Recent Stock Data"
    )

    st.dataframe(
        data.tail(10),
        use_container_width=True
    )


    # ========================================================
    # CLOSING PRICE
    # ========================================================

    st.header(
        "📈 Historical Closing Price"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        data.index,
        data["Close"],
        label="Closing Price"
    )

    ax.set_title(
        f"{stock} Closing Price"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Price ($)"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


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


    st.header(
        "📊 Moving Average Analysis"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        data.index,
        data["Close"],
        label="Close"
    )

    ax.plot(
        data.index,
        data["MA50"],
        label="MA50"
    )

    ax.plot(
        data.index,
        data["MA200"],
        label="MA200"
    )

    ax.set_title(
        f"{stock} Moving Averages"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Price ($)"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # RSI
    # ========================================================

    data["RSI"] = calculate_rsi(
        data["Close"]
    )

    st.header(
        "📉 Relative Strength Index"
    )

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(
        data.index,
        data["RSI"],
        label="RSI"
    )

    ax.axhline(
        70,
        linestyle="--",
        label="Overbought"
    )

    ax.axhline(
        30,
        linestyle="--",
        label="Oversold"
    )

    ax.set_title(
        f"{stock} RSI"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "RSI"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # VOLATILITY
    # ========================================================

    data["Volatility"] = (
        data["Close"]
        .pct_change()
        .rolling(20)
        .std()
    )

    st.header(
        "📊 Market Volatility"
    )

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(
        data.index,
        data["Volatility"],
        label="20-Day Volatility"
    )

    ax.set_title(
        f"{stock} Market Volatility"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Volatility"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # PREPARE DATA FOR LSTM
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
            "Error applying the saved scaler."
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

    X = np.array(X)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )


    # ========================================================
    # LSTM PREDICTIONS
    # ========================================================

    try:

        predictions_scaled = model.predict(
            X,
            verbose=0
        )

        predictions = (
            scaler.inverse_transform(
                predictions_scaled
            )
        )

        predictions = (
            predictions.flatten()
        )

    except Exception as error:

        st.error(
            "Error generating LSTM predictions."
        )

        st.code(
            str(error)
        )

        return


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    valid = data.iloc[
        sequence_length:
    ].copy()

    valid["Predicted"] = predictions


    st.header(
        "🤖 Actual vs Predicted Prices"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        valid.index,
        valid["Close"],
        label="Actual"
    )

    ax.plot(
        valid.index,
        valid["Predicted"],
        label="LSTM Predicted"
    )

    ax.set_title(
        f"{stock} Actual vs LSTM Predicted Prices"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Price ($)"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    rmse = math.sqrt(
        mean_squared_error(
            valid["Close"],
            valid["Predicted"]
        )
    )

    mae = mean_absolute_error(
        valid["Close"],
        valid["Predicted"]
    )


    st.header(
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
    # NEXT DAY PREDICTION
    # ========================================================

    last_60 = scaled_data[
        -sequence_length:
    ]

    last_60 = last_60.reshape(
        1,
        sequence_length,
        1
    )


    try:

        next_scaled = model.predict(
            last_60,
            verbose=0
        )

        next_price = (
            scaler.inverse_transform(
                next_scaled
            )
        )

        next_price = float(
            next_price[0][0]
        )

    except Exception as error:

        st.error(
            "Unable to generate next-day prediction."
        )

        st.code(
            str(error)
        )

        return


    predicted_change = (
        next_price -
        current_price
    )

    predicted_change_percent = (
        predicted_change /
        current_price
    ) * 100


    st.header(
        "🔮 Next-Day Price Prediction"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            f"${next_price:.2f}"
        )

    with col2:

        st.metric(
            "Expected Change",
            f"${predicted_change:.2f}"
        )

    with col3:

        st.metric(
            "Expected Change %",
            f"{predicted_change_percent:.2f}%"
        )


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    results = valid[
        [
            "Close",
            "Predicted"
        ]
    ].copy()

    results["Difference"] = (
        results["Predicted"] -
        results["Close"]
    )

    csv = results.to_csv(
        index=True
    ).encode("utf-8")


    st.download_button(
        "📥 Download Prediction Results",
        data=csv,
        file_name=f"{stock}_predictions.csv",
        mime="text/csv"
    )


    # ========================================================
    # NEWS & SENTIMENT
    # ========================================================

    st.header(
        "📰 Real-Time News & Sentiment Analysis"
    )


    # --------------------------------------------------------
    # READ NEWS API KEY
    # --------------------------------------------------------

    if "NEWS_API_KEY" not in st.secrets:

        st.warning(
            """
            NEWS_API_KEY is not configured.

            Add your NewsAPI key to Streamlit Secrets
            to enable news sentiment analysis.
            """
        )

    else:

        try:

            news_api_key = st.secrets[
                "NEWS_API_KEY"
            ]

            newsapi = NewsApiClient(
                api_key=news_api_key
            )


            articles = (
                newsapi.get_everything(

                    q=(
                        f"{stock} OR "
                        f"{stock} stock OR "
                        "financial markets OR "
                        "economy OR "
                        "inflation"
                    ),

                    language="en",

                    sort_by="publishedAt",

                    page_size=8
                )
            )


            article_list = articles.get(
                "articles",
                []
            )


            if not article_list:

                st.info(
                    "No recent news articles found."
                )

            else:

                positive = 0
                negative = 0
                neutral = 0


                for article in article_list:

                    title = article.get(
                        "title",
                        "No title"
                    )

                    description = article.get(
                        "description",
                        ""
                    )

                    url = article.get(
                        "url",
                        ""
                    )

                    source = article.get(
                        "source",
                        {}
                    ).get(
                        "name",
                        "Unknown"
                    )


                    sentiment, score = (
                        analyze_sentiment(
                            title
                        )
                    )


                    if sentiment.startswith(
                        "Positive"
                    ):

                        positive += 1

                    elif sentiment.startswith(
                        "Negative"
                    ):

                        negative += 1

                    else:

                        neutral += 1


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
                        f"**Polarity Score:** "
                        f"{score:.2f}"
                    )


                    if url:

                        st.markdown(
                            f"[Read Full Article]({url})"
                        )


                    st.markdown(
                        "---"
                    )


                # ------------------------------------------------
                # SENTIMENT SUMMARY
                # ------------------------------------------------

                st.subheader(
                    "📊 News Sentiment Summary"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Positive",
                        positive
                    )

                with c2:

                    st.metric(
                        "Negative",
                        negative
                    )

                with c3:

                    st.metric(
                        "Neutral",
                        neutral
                    )


        except Exception as error:

            st.error(
                f"News API error: {error}"
            )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.caption(
        "AI Stock Price Prediction System | "
        "LSTM Deep Learning + Real-Time Market Data + "
        "News Sentiment"
    )


# ============================================================
# RUN APP
# ============================================================

main()
