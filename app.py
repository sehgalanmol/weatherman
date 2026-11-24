import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import joblib


def load_models(model_path="./"):
    """Load previously saved Prophet models."""
    models = {
        'temp': joblib.load(f"{model_path}timeseries_temp.joblib"),
        'rhum': joblib.load(f"{model_path}timeseries_prcp.joblib"),
        'prcp': joblib.load(f"{model_path}timeseries_rhum.joblib"),
    }
    print("Models loaded successfully.")
    return models

if __name__ == "__main__":
    models = load_models()

def get_last_known_values_by_datetime(city, input_datetime, data):
    city_data = data[data['city'] == city]
    last_known_data = city_data[city_data['time'] <= input_datetime].sort_values(by='time', ascending=False).head(1)
    if last_known_data.empty:
        raise ValueError(f"No data available for {city} at or before {input_datetime}")
    return last_known_data.iloc[0]

def make_prediction(models, input_datetime, city, last_known_values):
    #next 3 days, 6-hour intervals
    future_dates = pd.date_range(start=input_datetime, periods=13, freq='6H')
    future_data = pd.DataFrame({'ds': future_dates, 'city': [city] * len(future_dates)})
    predictions = {}
    #temp
    temp_model = models['temp']
    temp_forecast = temp_model.predict(future_data[['ds', 'city']])
    predictions['temp'] = temp_forecast[['ds', 'yhat']]  
    
    #rhum
    rhum_model = models['rhum']
    rhum_forecast = rhum_model.predict(future_data[['ds', 'city']])
    predictions['rhum'] = rhum_forecast[['ds', 'yhat']]  
    
    #prcp
    prcp_model = models['prcp']
    prcp_forecast = prcp_model.predict(future_data[['ds', 'city']])
    predictions['prcp'] = prcp_forecast[['ds', 'yhat']]  
    
    return predictions


def plot_predictions(predictions, var):
    
    plt.figure(figsize=(10, 6))
    plt.plot(predictions[var]['ds'], predictions[var]['yhat'], label=f'{var} Prediction', marker='o')
    plt.title(f'{var.capitalize()} Prediction Over the Next 3 Days')
    plt.xlabel('Date')
    plt.ylabel(f'{var.capitalize()}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)


st.title("Weather Prediction App")
data = pd.read_csv('final1_small_data.csv')  
data['time'] = pd.to_datetime(data['time'], format='%d-%m-%Y %H:%M')
models = load_models()  
date_input = st.date_input("Select Date", datetime.now())
time_input = st.time_input("Select Time", datetime.now().time())
city_selection = st.selectbox("Select City", ["Delhi", "Kolkata"])  
city_map = {
    "Delhi": 1,
    "Kolkata": 0
}
city_input = city_map[city_selection] 
input_datetime = datetime.combine(date_input, time_input)

if st.button("Get Predictions"):
    try:
        last_known_values = get_last_known_values_by_datetime(city_input, input_datetime, data)
        predictions = make_prediction(models, input_datetime, city_input, last_known_values)
        for var in predictions.keys():
            plot_predictions(predictions, var)
    except ValueError as e:
        st.error(str(e))
