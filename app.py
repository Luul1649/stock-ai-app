import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Price Prediction using LSTM")

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_model_scaler():
    model = load_model("lstm_model_cleaned.h5")
    scaler = pickle.load(open("scaler.pkl","rb"))
    return model, scaler

model, scaler = load_model_scaler()

# -----------------------------
# User Input
# -----------------------------
stock = st.text_input("Enter Stock Symbol", "GOOG")

start = "2015-01-01"
end = "2024-12-31"

# -----------------------------
# Download Stock Data
# -----------------------------
data = yf.download(stock, start, end)

st.subheader("Stock Data")
st.write(data.tail())

# -----------------------------
# Closing Price Chart
# -----------------------------
st.subheader("Closing Price Chart")
st.line_chart(data['Close'])

# -----------------------------
# Prepare Data
# -----------------------------
timesteps = 100
future_days = 30

if len(data) < timesteps:
    st.error("Not enough data for prediction (need at least 100 records)")
    st.stop()

close_data = data[['Close']]

scaled_data = scaler.transform(close_data)

# -----------------------------
# Prepare Test Data
# -----------------------------
x_test = []
y_test = []

for i in range(timesteps, scaled_data.shape[0]):
    x_test.append(scaled_data[i-timesteps:i])
    y_test.append(scaled_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# -----------------------------
# Model Prediction
# -----------------------------
predictions = model.predict(x_test)

predicted_prices = scaler.inverse_transform(predictions)

real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# -----------------------------
# Actual vs Predicted
# -----------------------------
st.subheader("Actual vs Predicted Prices")

result = pd.DataFrame({
    "Actual Price": real_prices.flatten(),
    "Predicted Price": predicted_prices.flatten()
})

st.line_chart(result)

# -----------------------------
# Future Prediction
# -----------------------------
st.subheader("Next 30 Days Prediction")

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

# -----------------------------
# Create Future Dates
# -----------------------------
last_date = data.index[-1]

future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_predictions
})

future_df.set_index("Date", inplace=True)

st.line_chart(future_df)
