import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from newsapi import NewsApiClient
from textblob import TextBlob

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📈 AI Stock Price Prediction & Market Sentiment Dashboard")

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

@st.cache_resource
def load_model_scaler():
    model = load_model("lstm_model_cleaned.h5")
    scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

model, scaler = load_model_scaler()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("Stock Settings")

stocks = st.sidebar.multiselect(
    "Select Stocks",
    ["GOOG","AAPL","TSLA","AMZN","MSFT"],
    default=["GOOG"]
)

start = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

future_days = st.sidebar.slider("Future Prediction Days", 7, 90, 30)

# --------------------------------------------------
# DOWNLOAD STOCK DATA
# --------------------------------------------------
stock = st.text_input("Enter Stock Symbol", "GOOG")
data = yf.download(stock, start=start, end=end)

# Use Close column as DataFrame
close_prices = pd.DataFrame(data['Close'])
close_prices.columns = ['Close']  # rename column to 'Close' for consistency

st.line_chart(close_prices)
# --------------------------------------------------
# STOCK PRICE VISUALIZATION
# --------------------------------------------------

st.subheader("📊 Stock Price Trend")

st.line_chart(close_prices)

# --------------------------------------------------
# CANDLESTICK CHART
# --------------------------------------------------

if len(stocks) == 1:

    st.subheader("🕯 Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    st.plotly_chart(fig)

# --------------------------------------------------
# VOLATILITY ANALYSIS
# --------------------------------------------------

st.subheader("📉 Volatility Analysis")

returns = close_prices.pct_change()

volatility = returns.std() * np.sqrt(252)

st.write("Annual Volatility")

st.write(volatility)

st.line_chart(returns)

# --------------------------------------------------
# LSTM MODEL PREDICTION
# --------------------------------------------------

st.subheader("🤖 AI Model Prediction")

timesteps = 100

close_data = close_prices

if len(close_data) < timesteps:
    st.error("Not enough data for prediction (need at least 100 rows)")
    st.stop()

scaled_data = scaler.transform(close_data)

x_test = []
y_test = []

for i in range(timesteps, scaled_data.shape[0]):
    x_test.append(scaled_data[i-timesteps:i])
    y_test.append(scaled_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

predictions = model.predict(x_test)

predicted_prices = scaler.inverse_transform(predictions)

real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.subheader("📊 Model Performance")

rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
mae = mean_absolute_error(real_prices, predicted_prices)

col1, col2 = st.columns(2)

col1.metric("RMSE", round(rmse,2))
col2.metric("MAE", round(mae,2))

# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

st.subheader("Actual vs Predicted Prices")

result = pd.DataFrame({
    "Actual": real_prices.flatten(),
    "Predicted": predicted_prices.flatten()
})

st.line_chart(result)

# --------------------------------------------------
# FUTURE PREDICTION
# --------------------------------------------------

st.subheader("🔮 Future Stock Price Prediction")

last_days = close_data.tail(timesteps).values

future_predictions = []

input_seq = last_days.copy()

for i in range(future_days):

    scaled = scaler.transform(input_seq)

    X = []
    X.append(scaled)

    X = np.array(X)

    pred = model.predict(X)

    pred_price = scaler.inverse_transform(pred)

    future_predictions.append(pred_price[0][0])

    input_seq = np.append(input_seq[1:], pred_price, axis=0)

last_date = close_data.index[-1]

future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_predictions
})

future_df.set_index("Date", inplace=True)

st.line_chart(future_df)

# --------------------------------------------------
# NEWS SENTIMENT ANALYSIS
# --------------------------------------------------

st.subheader("📰 Political & Market Sentiment")

newsapi = NewsApiClient(api_key="YOUR_NEWSAPI_KEY")

news = newsapi.get_everything(
    q=stocks[0],
    language="en",
    sort_by="publishedAt",
    page_size=10
)

sentiments = []

for article in news["articles"]:

    title = article["title"]

    analysis = TextBlob(title)

    sentiment = analysis.sentiment.polarity

    sentiments.append(sentiment)

    st.write("📰", title)

avg_sentiment = np.mean(sentiments)

st.write("Average Sentiment Score:", round(avg_sentiment,3))

if avg_sentiment > 0.1:
    st.success("Market Sentiment Positive 📈")

elif avg_sentiment < -0.1:
    st.error("Market Sentiment Negative 📉")

else:
    st.warning("Market Sentiment Neutral ⚖️")


