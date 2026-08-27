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

st.sidebar.write("Examples:")
st.sidebar.write("AAPL • TSLA • NVDA • MSFT • AMZN")

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Refresh Stock Data"):
    st.cache_data.clear()
    st.rerun()


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
# FETCH STOCK DATA FROM YAHOO FINANCE
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data(ticker):

    ticker = ticker.strip().upper()

    try:

        data = yf.download(
            tickers=ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        # ----------------------------------------------------
        # CHECK DATA
        # ----------------------------------------------------

        if data is None or data.empty:

            return pd.DataFrame(), (
                f"Yahoo Finance returned no data for {ticker}."
            )

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

        # ----------------------------------------------------
        # CHECK MISSING COLUMNS
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            return pd.DataFrame(), (
                "Yahoo Finance is missing these columns: "
                f"{missing_columns}"
            )

        # ----------------------------------------------------
        # KEEP REQUIRED COLUMNS
        # ----------------------------------------------------

        data = data[
            required_columns
        ].copy()

        # ----------------------------------------------------
        # CONVERT DATA TO NUMERIC
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

        # ----------------------------------------------------
        # MINIMUM DATA CHECK
        # ----------------------------------------------------

        if len(data) < 65:

            return pd.DataFrame(), (
                f"Only {len(data)} records were returned. "
                "At least 65 records are required "
                "for the LSTM model."
            )

        return data, None

    except Exception as error:

        return pd.DataFrame(), (
            f"Yahoo Finance error: {error}"
        )


# ============================================================
# RSI CALCULATION
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
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text):

    if not text:

        return (
            "Neutral ➖",
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
            Make sure the following files are in
            your GitHub repository:

            • lstm_model_cleaned.h5
            • scaler.pkl
            """
        )

        return


    # ========================================================
    # FETCH STOCK DATA
    # ========================================================

    with st.spinner(
        f"Loading {stock} market data..."
    ):

        data, data_error = fetch_stock_data(
            stock
        )


    # ========================================================
    # DATA VALIDATION
    # ========================================================

    if data.empty:

        st.error(
            f"❌ Could not retrieve stock data for **{stock}**."
        )

        if data_error:

            st.warning(
                data_error
            )

        st.info(
            """
            Check the stock symbol and try the
            **Refresh Stock Data** button.

            Examples:

            AAPL
            TSLA
            NVDA
            MSFT
            AMZN
            """
        )

        return


    # ========================================================
    # CURRENT MARKET VALUES
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
    # MARKET OVERVIEW
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
    # RECENT STOCK DATA
    # ========================================================

    st.header(
        "📋 Recent Stock Data"
    )

    display_data = data.tail(
        10
    ).copy()

    st.dataframe(
        display_data,
        use_container_width=True
    )


    # ========================================================
    # HISTORICAL CLOSING PRICE
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
        f"{stock} Historical Closing Price"
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
        f"{stock} Moving Average Analysis"
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
        "📉 Relative Strength Index (RSI)"
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
        label="Overbought (70)"
    )

    ax.axhline(
        30,
        linestyle="--",
        label="Oversold (30)"
    )

    ax.set_title(
        f"{stock} Relative Strength Index"
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
    # MARKET VOLATILITY
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
            "❌ Error applying the saved scaler."
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

        sequence = (
            scaled_data[
                i - sequence_length:i,
                0
            ]
        )

        X.append(
            sequence
        )

    X = np.array(X)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )


    # ========================================================
    # GENERATE LSTM PREDICTIONS
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
            "❌ Error generating LSTM predictions."
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
        label="Actual Price"
    )

    ax.plot(
        valid.index,
        valid["Predicted"],
        label="LSTM Predicted Price"
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
        "📊 LSTM Model Performance"
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

        next_prediction = (
            scaler.inverse_transform(
                next_scaled
            )
        )

        next_price = float(
            next_prediction[0][0]
        )

    except Exception as error:

        st.error(
            "❌ Unable to generate next-day prediction."
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


    # ========================================================
    # NEXT DAY PREDICTION DISPLAY
    # ========================================================

    st.header(
        "🔮 Next-Day Stock Price Prediction"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted Price",
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
    # PREDICTION INTERPRETATION
    # ========================================================

    if predicted_change_percent > 0:

        st.success(
            f"""
            📈 The LSTM model predicts an increase
            from ${current_price:.2f} to
            approximately ${next_price:.2f}.
            """
        )

    elif predicted_change_percent < 0:

        st.warning(
            f"""
            📉 The LSTM model predicts a decrease
            from ${current_price:.2f} to
            approximately ${next_price:.2f}.
            """
        )

    else:

        st.info(
            "The model predicts little or no price change."
        )


    # ========================================================
    # DOWNLOAD PREDICTION RESULTS
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

    csv_data = results.to_csv(
        index=True
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv_data,
        file_name=f"{stock}_predictions.csv",
        mime="text/csv"
    )


    # ========================================================
    # NEWS API
    # ========================================================

    st.header(
        "📰 Real-Time News & Sentiment Analysis"
    )


    if "NEWS_API_KEY" not in st.secrets:

        st.warning(
            """
            ⚠️ NEWS_API_KEY has not been configured.

            Add your NewsAPI key under:

            Streamlit Cloud →
            App Settings →
            Secrets
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


            # ------------------------------------------------
            # SEARCH NEWS
            # ------------------------------------------------

            articles_response = (
                newsapi.get_everything(

                    q=(
                        f'"{stock}" OR '
                        f'"{stock} stock" OR '
                        '"financial markets" OR '
                        '"global economy" OR '
                        '"inflation"'
                    ),

                    language="en",

                    sort_by="publishedAt",

                    page_size=8
                )
            )


            articles = articles_response.get(
                "articles",
                []
            )


            # ------------------------------------------------
            # CHECK ARTICLES
            # ------------------------------------------------

            if not articles:

                st.info(
                    "No recent news articles were found."
                )

            else:

                positive_count = 0
                negative_count = 0
                neutral_count = 0


                # ============================================
                # DISPLAY ARTICLES
                # ============================================

                for article in articles:

                    title = article.get(
                        "title",
                        "No title available"
                    )

                    description = article.get(
                        "description",
                        ""
                    )

                    article_url = article.get(
                        "url",
                        ""
                    )

                    source_info = article.get(
                        "source",
                        {}
                    )

                    source_name = source_info.get(
                        "name",
                        "Unknown source"
                    )


                    # ------------------------------------------------
                    # SENTIMENT
                    # ------------------------------------------------

                    sentiment, polarity = (
                        analyze_sentiment(
                            title
                        )
                    )


                    if sentiment.startswith(
                        "Positive"
                    ):

                        positive_count += 1

                    elif sentiment.startswith(
                        "Negative"
                    ):

                        negative_count += 1

                    else:

                        neutral_count += 1


                    # ------------------------------------------------
                    # ARTICLE DISPLAY
                    # ------------------------------------------------

                    st.markdown(
                        f"### {title}"
                    )

                    if description:

                        st.write(
                            description
                        )

                    st.write(
                        f"**Source:** {source_name}"
                    )

                    st.write(
                        f"**Sentiment:** {sentiment}"
                    )

                    st.write(
                        f"**Polarity Score:** "
                        f"{polarity:.2f}"
                    )

                    if article_url:

                        st.markdown(
                            f"[Read Full Article]({article_url})"
                        )

                    st.markdown(
                        "---"
                    )


                # ============================================
                # SENTIMENT SUMMARY
                # ============================================

                st.subheader(
                    "📊 News Sentiment Summary"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Positive News",
                        positive_count
                    )

                with col2:

                    st.metric(
                        "Negative News",
                        negative_count
                    )

                with col3:

                    st.metric(
                        "Neutral News",
                        neutral_count
                    )


        except Exception as error:

            st.error(
                f"❌ News API error: {error}"
            )


    # ========================================================
    # PROJECT DISCLAIMER
    # ========================================================

    st.markdown("---")

    st.info(
        """
        **Project Disclaimer**

        This application is an academic AI stock prediction
        project. Predictions are generated by an LSTM deep
        learning model using historical market data.

        The predictions and sentiment analysis are for
        educational and research purposes only and should
        not be considered financial advice.
        """
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
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
