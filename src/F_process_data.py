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

from utils import create_violin_plot, get_forecast_hourly_weather, merge_csv_files

url = 'https://archive-api.open-meteo.com/v1/archive?latitude=50.88&longitude=4.70&start_date=2022-01-01&end_date=2022-12-31&timezone=Europe%2FBerlin&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,pressure_msl,surface_pressure,precipitation,snowfall,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,windspeed_10m,winddirection_10m,windgusts_10m'
weather_data = get_forecast_hourly_weather(url)


## rename the column
weather_data.columns = [re.sub(' \(.*\)', '', col) for col in weather_data.columns]

# Format time stamp
weather_data['time'] = pd.to_datetime(weather_data['time'])
weather_data['date'] = weather_data['time'].dt.date
weather_data['hour'] = weather_data['time'].dt.hour
weather_data['month'] = weather_data['time'].dt.month
weather_data['weekday'] = weather_data['time'].dt.strftime('%a')

weather_data.to_csv('../data/processed_weather_data.csv')


"""
 Data Processing: File 40
"""
file40 = merge_csv_files('../data/file40')
### we would delete unit column because of same values
# drop all _unit columns
cols_to_drop = [col for col in file40.columns if col.endswith('unit')]
file40.drop(cols_to_drop, axis=1, inplace=True)

# rename columns
file40.rename(columns={'description': 'location', '#object_id': 'object_id'}, inplace=True)

# Convert the 'result_timestamp' column to a datetime data type
file40['result_timestamp'] = pd.to_datetime(file40['result_timestamp'])
file40['date'] = file40['result_timestamp'].dt.date
file40['hour'] = file40['result_timestamp'].dt.hour
file40['month'] = file40['result_timestamp'].dt.month
file40['weekday'] = file40['result_timestamp'].dt.strftime('%a')

file40.to_csv('../data/processed_file40_data.csv')


"""
 Data Processing: File 41
"""
# Merge files
file41 = merge_csv_files('../data/file41')
# drop unncessary cols
cols_to_drop = [
	'noise_event_laeq_model_id_unit',
	'noise_event_laeq_model_id',
	'noise_event_laeq_primary_detected_certainty_unit',
	'noise_event_laeq_primary_detected_class_unit',
]

file41.drop(cols_to_drop, axis=1, inplace=True)

# rename cols
file41.columns = [
	'object_id',
	'location',
	'result_timestamp',
	'noise_event_certainty',
	'noise_event',
]

# remove the noise_event that are unsupported
file41 = file41.loc[file41.noise_event != 'Unsupported']

# extract from timestamp
file41['result_timestamp'] = pd.to_datetime(file41['result_timestamp'])
file41['time'] = file41['result_timestamp'].dt.time
file41['date'] = file41['result_timestamp'].dt.date
file41['hour'] = file41['result_timestamp'].dt.hour
file41['month'] = file41['result_timestamp'].dt.month
file41['weekday'] = file41['result_timestamp'].dt.strftime('%a')

# Only adding datapoints with certanity of 85%+
file41 = pd.DataFrame(file41.loc[file41['noise_event_certainty'] > 85].reset_index())
# Dropping undefined columns
file41 = file41.dropna(subset=['noise_event'])
# Pivot table to transform into counting hourly types of noise events
file41_piv = (
	pd.pivot_table(
		file41,
		index=['object_id', 'date', 'hour', 'month', 'weekday'],
		columns=['noise_event'],
		aggfunc='count',
	)
	.xs('location', level=0, axis=1)
	.reset_index()
)
file41_piv.fillna(0, inplace=True)

file41_piv.to_csv('../data/processed_file41_data.csv')

"""
 Data Processing: File 42
"""
# folder containing the full data
main_folder = 's3://city-noise-predictor/export_42_full/'

# list of months and their numerical codes used in the data structure
months = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
num_codes = range(42, 53)
num_codes = [str(num) for num in num_codes]
month_codes = {month: code for month, code in zip(months, num_codes)}
month_codes['Feb'] = '42'

# dictionary of location codes
object_id_dict = {
	255439: 'MP 01: Naamsestraat 35 Maxim',
	255440: 'MP 02: Naamsestraat 57 Xior',
	255441: 'MP 03: Naamsestraat 62 Taste',
	303910: 'MP 04: His & Hears',
	255442: 'MP 05: Calvariekapel KU Leuven',
	255443: 'MP 06: Parkstraat 2 La Filosovia',
	255444: 'MP 07: Naamsestraat 81',
	280324: 'MP08bis - Vrijthof',
}

# File name list, the stuff that comes after csv_results_[monthcode]
file_name_exts = [
	'_255439_mp-01-naamsestraat-35-maxim.csv',
	'_255440_mp-02-naamsestraat-57-xior.csv',
	'_255441_mp-03-naamsestraat-62-taste.csv',
	'_255442_mp-05-calvariekapel-ku-leuven.csv',
	'_255443_mp-06-parkstraat-2-la-filosovia.csv',
	'_255444_mp-07-naamsestraat-81.csv',
	'_255445_mp-08-kiosk-stadspark.csv',
	'_280324_mp08bis---vrijthof.csv',
	'_303910_mp-04-his-hears.csv',
]


# function to give the list of files for a month's folder
def filelist(monthstring):
	filelist = []
	for file in file_name_exts:
		filelist.append(
			main_folder + monthstring + '/csv_results_' + month_codes[monthstring] + file
		)
	return filelist


print(filelist('March'))


def gather_file_42():
	"""
	Gathers and processes all files in the file42 directory for specified months.
	Args:
	months (list): List of months to process, e.g., ['01', '02', '03'] for January, February, March.
	Returns:
	pandas.DataFrame: A DataFrame containing the processed data from all files.
	"""
	concatenated_df = pd.DataFrame()

	for month in months:
		file_list = filelist(month)

		for file in file_list:
			df_resampled = process_file(file)
			# concatenate
			concatenated_df = pd.concat([concatenated_df, df_resampled])
			concatenated_df = concatenated_df[
				[
					'result_timestamp',
					'object_id',
					'lamax',
					'laeq',
					'lceq',
					'lcpeak',
					'date',
					'hour',
					'weekday',
					'month',
					'location',
				]
			]
	return concatenated_df


file42 = gather_file_42()
file42.to_csv('../data/processed_file42.csv')
