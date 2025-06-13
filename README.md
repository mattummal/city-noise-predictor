# City Noise Predictor: a Machine Learning WebApp


The objective is to develop an application that predicts the noise level in a particular city. The prediction model will be based on forecast weather and air quality data. By utilising machine learning models, we aim to provide valuable insights into the noise levels in the city, enabling residents and authorities to better understand and manage noise pollution.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mda-woise.streamlit.app/)


## 🌦 Data collection

* Air quality data:
  * Historical: scraped using [OpenWeatherMap](https://openweathermap.org/api/air-pollution) API
  * Forecast: scraped using [Open-Meteo forecast air quality API](https://open-meteo.com/en/docs/air-quality-api)
* Weather data:
  * Historical: scraped using [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) with [query](https://archive-api.open-meteo.com/v1/archive?latitude=50.88&longitude=4.70&start_date=2022-01-01&end_date=2022-12-31&timezone=Europe%2FBerlin&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,pressure_msl,surface_pressure,precipitation,snowfall,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,windspeed_10m,winddirection_10m,windgusts_10m&format=csv)
  * Forecast: scraped using [Open-Meteo forecast weather API](https://open-meteo.com/en/docs) with [query](https://api.open-meteo.com/v1/forecast?latitude=50.88&longitude=4.70&timezone=Europe%2FBerlin&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,pressure_msl,surface_pressure,precipitation,snowfall,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,windspeed_10m,winddirection_10m,windgusts_10m)

## 📚 File organization

```
📦
├─ README.md
├─ __pycache__
├─ conda_requirements.txt
├─ pip_requirements.txt
├─ app
│  ├─ .streamlit
│  │  └─ config.toml
│  ├─ __pycache__
│  │  ├─ historical_noise.cpython-39.pyc
│  │  ├─ prediction_noise.cpython-39.pyc
│  │  └─ weather.cpython-39.pyc
│  ├─ historical_noise.py
│  ├─ main.py
│  ├─ prediction_noise.py
│  ├─ requirements.txt
│  ├─ weather.py
├─ data
│  ├─ file40
│  ├─ file41.csv
│  ├─ file41
│  ├─ processed_air_quality_data.csv
│  ├─ processed_file40_data.csv
│  ├─ processed_file41_data.csv
│  ├─ processed_file42_data.csv
│  └─ processed_weather_data.csv
├─ model
│  ├─ model_noise_level_file40
│  ├─ model_noise_level_file42
│  └─ noise_types
└─ notebook
   ├─ 1_EDA.ipynb
   ├─ 2_scrape_and_process_data.ipynb
   ├─ 3_model_predict_noise_level_file40.ipynb
   ├─ 4_model_predict_noise_level_file42.ipynb
   ├─ 5_model_predict_noise_types.ipynb
   └─ 6_test_predictions.ipynb
```
