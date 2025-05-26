import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load models and scalers
@st.cache_resource
def load_models():
    rf_model = joblib.load('best_random_forest_model.joblib')
    lstm_model = load_model('lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    scaler_lstm = joblib.load('scaler_lstm.pkl')
    lstm_features = joblib.load('lstm_features.pkl')
    return rf_model, lstm_model, scaler, scaler_lstm, lstm_features

rf_model, lstm_model, scaler, scaler_lstm, lstm_features = load_models()

# App title
st.title("🏠 Household Power Consumption Predictor")
st.write("Predict the next hour's global active power consumption using ML models")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Create input fields for all features
hour = st.sidebar.slider("Hour of day", 0, 23, 12)
dayofweek = st.sidebar.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=1)
month = st.sidebar.slider("Month", 1, 12, 12)
dayofyear = st.sidebar.slider("Day of year", 1, 365, 351)

# Convert day of week to numerical
day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, 
           "Friday":4, "Saturday":5, "Sunday":6}
dayofweek_num = day_map[dayofweek]

# Main power parameters
st.sidebar.subheader("Current Power Metrics")
global_active_power = st.sidebar.number_input("Global Active Power (kW)", min_value=0.0, value=2.5, step=0.1)
global_reactive_power = st.sidebar.number_input("Global Reactive Power (kW)", min_value=0.0, value=0.12, step=0.01)
voltage = st.sidebar.number_input("Voltage (V)", min_value=220.0, value=240.1, step=1.0)
global_intensity = st.sidebar.number_input("Global Intensity (A)", min_value=0.0, value=10.5, step=0.5)

# Sub-metering
st.sidebar.subheader("Sub-metering (Wh)")
sub_metering_1 = st.sidebar.number_input("Kitchen", min_value=0, value=0)
sub_metering_2 = st.sidebar.number_input("Laundry", min_value=0, value=1)
sub_metering_3 = st.sidebar.number_input("HVAC/Water Heater", min_value=0, value=17)

# Lag features
st.sidebar.subheader("Historical Data")
lag1 = st.sidebar.number_input("1 Hour Ago (kW)", min_value=0.0, value=2.4, step=0.1)
lag2 = st.sidebar.number_input("2 Hours Ago (kW)", min_value=0.0, value=2.3, step=0.1)
lag3 = st.sidebar.number_input("3 Hours Ago (kW)", min_value=0.0, value=2.2, step=0.1)
rolling_mean = st.sidebar.number_input("24h Average (kW)", min_value=0.0, value=2.35, step=0.1)
rolling_std = st.sidebar.number_input("24h Std Dev (kW)", min_value=0.0, value=0.15, step=0.01)

# Create input DataFrame
input_data = pd.DataFrame({
    'global_active_power': [global_active_power],
    'global_reactive_power': [global_reactive_power],
    'voltage': [voltage],
    'global_intensity': [global_intensity],
    'sub_metering_1': [sub_metering_1],
    'sub_metering_2': [sub_metering_2],
    'sub_metering_3': [sub_metering_3],
    'Hour': [hour],
    'DayOfWeek': [dayofweek_num],
    'Month': [month],
    'DayOfYear': [dayofyear],
    'global_active_power_lag1': [lag1],
    'global_active_power_lag2': [lag2],
    'global_active_power_lag3': [lag3],
    'global_active_power_rolling_mean_24h': [rolling_mean],
    'global_active_power_rolling_std_24h': [rolling_std]
})

# Prediction function
def make_predictions(input_df):
    # Random Forest prediction
    rf_input = scaler.transform(input_df)
    rf_pred = rf_model.predict(rf_input)[0]
    
    # LSTM prediction (using current values as dummy sequence)
    lstm_input = input_df[lstm_features]
    sequence = np.tile(scaler_lstm.transform(lstm_input), (24, 1))
    sequence = sequence.reshape(1, 24, len(lstm_features))
    lstm_pred_scaled = lstm_model.predict(sequence)[0][0]
    power_min = scaler_lstm.data_min_[0]
    power_max = scaler_lstm.data_max_[0]
    lstm_pred = lstm_pred_scaled * (power_max - power_min) + power_min
    
    return rf_pred, lstm_pred

# Display predictions
if st.button("Predict Next Hour's Consumption"):
    rf_pred, lstm_pred = make_predictions(input_data)
    
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Random Forest Prediction", f"{rf_pred:.2f} kW")
    
    with col2:
        st.metric("LSTM Prediction", f"{lstm_pred:.2f} kW")
    
    # Show input data
    st.subheader("Input Parameters Used")
    st.dataframe(input_data)

# For better LSTM predictions, you would need actual 24h sequence
st.info("Note: The LSTM prediction uses current values as a dummy sequence. For accurate results, provide actual 24h historical data.")
