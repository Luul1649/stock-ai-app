import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Stock AI Predictor", layout="wide")

st.title("📊 LSTM Stock Price Prediction Dashboard")

# ------------------------------------
# Load Model and Scaler
# ------------------------------------
@st.cache_resource
def load_model_scaler():
    model = load_model("lstm_model_cleaned.h5")
    scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

model, scaler = load_model_scaler()

# ------------------------------------
# Sidebar Inputs
# ------------------------------------
st.sidebar.header("User Input")

stocks = st.sidebar.multiselect(
    "Select Stocks",
    ["GOOG","AAPL","TSLA","AMZN","MSFT"],
    default=["GOOG"]
)

start = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

future_days = st.sidebar.slider("Future Prediction Days", 7, 90, 30)

# ------------------------------------
# Download Data
# ------------------------------------
data = yf.download(stocks, start=start, end=end)

if len(stocks) == 1:
    data = data

else:
    data = data["Close"]

# ------------------------------------
# Stock Comparison
# ------------------------------------
st.subheader("📊 Stock Comparison")

if len(stocks) > 1:
    st.line_chart(data)
else:
    st.line_chart(data['Close'])

# ------------------------------------
# Candlestick Chart
# ------------------------------------
st.subheader("🕯 Candlestick Chart")

if len(stocks) == 1:

    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    st.plotly_chart(fig)

# ------------------------------------
# Volatility Analysis
# ------------------------------------
st.subheader("📉 Volatility Analysis")

returns = data['Close'].pct_change()

volatility = returns.std() * np.sqrt(252)

st.write("Annual Volatility:", volatility)

st.line_chart(returns)

# ------------------------------------
# LSTM Prediction
# ------------------------------------
timesteps = 100

if len(data) < timesteps:
    st.error("Not enough data for prediction")
    st.stop()

close_data = data[['Close']]

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

# ------------------------------------
# Evaluation Metrics
# ------------------------------------
st.subheader("📊 Model Performance")

rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
mae = mean_absolute_error(real_prices, predicted_prices)

col1, col2 = st.columns(2)

col1.metric("RMSE", round(rmse,2))
col2.metric("MAE", round(mae,2))

# ------------------------------------
# Actual vs Predicted
# ------------------------------------
st.subheader("📈 Actual vs Predicted Prices")

result = pd.DataFrame({
    "Actual": real_prices.flatten(),
    "Predicted": predicted_prices.flatten()
})

st.line_chart(result)

# ------------------------------------
# Future Prediction
# ------------------------------------
st.subheader(f"🔮 Next {future_days} Days Prediction")

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

last_date = data.index[-1]

future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_predictions
})

future_df.set_index("Date", inplace=True)

st.line_chart(future_df)
