import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pytz
import requests
import seaborn as sns
from matplotlib.dates import DateFormatter, MonthLocator

warnings.filterwarnings('ignore')
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from utils import create_violin_plot, merge_csv_files


# Get air quality data for Leuven
# Period: Jan 1, 2022 to Dec 31, 2022
api_key = 'ec722f11f234fb9a316e2580e6e2019e'
lat = 50.88
lon = 4.7
start_date = '1640995200'
end_date = '1672531200'

url = f'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_date}&end={end_date}&appid={api_key}'
# url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
# url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={api_key}"

response = requests.get(url)

if response.status_code == 200:
	data = response.json()
else:
	print('Error:', response.status_code)

# Historical
df = pd.DataFrame(data['list'])
df['dt'] = pd.to_datetime(df['dt'], unit='s')
components_df = pd.json_normalize(df['components'])
aqi_df = pd.json_normalize(df['main'])
df = pd.concat([df, components_df, aqi_df], axis=1)
df.drop(['components', 'main'], axis=1, inplace=True)


air_quality_data = df

# Format time stamp
air_quality_data['dt'] = pd.to_datetime(air_quality_data['dt'])
air_quality_data['date'] = air_quality_data['dt'].dt.date
air_quality_data['hour'] = air_quality_data['dt'].dt.hour
air_quality_data['month'] = air_quality_data['dt'].dt.month
air_quality_data['weekday'] = air_quality_data['dt'].dt.strftime('%a')
air_quality_data.to_csv('../data/processed_air_quality_data.csv')

url = 'https://telraam-api.net/v1/reports/traffic'
headers = {'X-Api-Key': 'Z9Qoyuy9Yf99nf2I0Myig4t0ftuUmY81ahBIVZH4'}

start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2023-01-01')

# define the duration for each loop (3 months)
loop_duration = pd.DateOffset(months=3)

# define the list of sensor IDs
counter_ids = [
	'9000000627',
	'347690',
	'9000000674',
	'9000000773',
	'347931',
	'347860',
	'9000000764',
	'9000000672',
	'347948',
	'9000001547',
	'347365',
	'349054',
	'9000000681',
]

full_data = pd.DataFrame()

# loop through the counter IDs
for counter_id in counter_ids:
	# retrieve data for each 3-month period
	current_date = start_date
	while current_date < end_date:
		loop_start_date = current_date
		loop_end_date = loop_start_date + loop_duration

		body = {
			'id': counter_id,
			'time_start': loop_start_date.strftime('%Y-%m-%d %H:%M:%SZ'),
			'time_end': loop_end_date.strftime('%Y-%m-%d %H:%M:%SZ'),
			'level': 'segments',
			'format': 'per-hour',
		}
		payload = str(body)

		response = requests.post(url, headers=headers, data=payload)
		json_data = response.json()
		loop_data = pd.DataFrame(json_data['report'])
		full_data = pd.concat([full_data, loop_data])

		# update the current date for the next loop
		current_date += loop_duration


location = {
	'9000000627': 'redingenhof',
	'347690': 'Kapucijnenvoer',
	'9000000674': 'Tiensestraat',
	'9000000773': 'Bondgenotenlaan',
	'347931': 'Vital Decostersstraat',
	'347860': 'BROUWERSSTRAAT',
	'9000000764': 'Fonteinstraat 137 b 301',
	'9000000672': 'Petermannenstraat',
	'347948': 'Ridderstraat',
	'9000001547': 'Jan Pieter Minckelersstraat',
	'347365': 'Dekenstraat',
	'349054': 'Pleinstraat',
	'9000000681': 'Bierbeekstraat',
}

full_data['location'] = full_data.segment_id.astype('str').map(location)
full_data.to_csv('../data/traffic_2022.csv', index=False)
