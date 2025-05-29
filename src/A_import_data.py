import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import seaborn as sns

from utils import merge_csv_files

folder_path = '../data/'
file_list_40 = [
	'csv_results_40_255439_mp-01-naamsestraat-35-maxim.csv',
	'csv_results_40_255440_mp-02-naamsestraat-57-xior.csv',
	'csv_results_40_255441_mp-03-naamsestraat-62-taste.csv',
	'csv_results_40_255442_mp-05-calvariekapel-ku-leuven.csv',
	'csv_results_40_255443_mp-06-parkstraat-2-la-filosovia.csv',
	'csv_results_40_255444_mp-07-naamsestraat-81.csv',
	'csv_results_40_255445_mp-08-kiosk-stadspark.csv',
	'csv_results_40_280324_mp08bis---vrijthof.csv',
	'csv_results_40_303910_mp-04-his-hears.csv',
]

file_list_41 = [
	'csv_results_41_255439_mp-01-naamsestraat-35-maxim.csv',
	'csv_results_41_255440_mp-02-naamsestraat-57-xior.csv',
	'csv_results_41_255441_mp-03-naamsestraat-62-taste.csv',
	'csv_results_41_255442_mp-05-calvariekapel-ku-leuven.csv',
	'csv_results_41_255443_mp-06-parkstraat-2-la-filosovia.csv',
	'csv_results_41_255444_mp-07-naamsestraat-81.csv',
	'csv_results_41_255445_mp-08-kiosk-stadspark.csv',
	'csv_results_41_280324_mp08bis---vrijthof.csv',
	'csv_results_41_303910_mp-04-his-hears.csv',
]

file_list_42 = [
	'csv_results_42_255439_mp-01-naamsestraat-35-maxim.csv',
	'csv_results_42_255440_mp-02-naamsestraat-57-xior.csv',
	'csv_results_42_255441_mp-03-naamsestraat-62-taste.csv',
	'csv_results_42_255442_mp-05-calvariekapel-ku-leuven.csv',
	'csv_results_42_255443_mp-06-parkstraat-2-la-filosovia.csv',
	'csv_results_42_255444_mp-07-naamsestraat-81.csv',
	'csv_results_42_255445_mp-08-kiosk-stadspark.csv',
	'csv_results_42_280324_mp08bis---vrijthof.csv',
	'csv_results_42_303910_mp-04-his-hears.csv',
]

# lots of files, takes a while
file40 = merge_csv_files(folder_path + '/export_40/', file_list_40)
file41 = merge_csv_files(folder_path + '/export_41/', file_list_41)
file42 = merge_csv_files(
	folder_path + '/export_42/', file_list_42
)  # Uses the incomplete, reduced data set


file_list_meteo = [
	'LC_2022Q1.csv',
	'LC_2022Q2.csv',
	'LC_2022Q3.csv',
	'LC_2022Q4.csv',
]

# lots of files, takes a while
meteo = merge_csv_files(folder_path + '/meteodata/', file_list_meteo, delim=',')

belgium_tz = pytz.timezone('Europe/Brussels')

# convert the 'dateutc' column to datetime objects and set the timezone to UTC
meteo['DATEUTC'] = pd.to_datetime(meteo['DATEUTC'], utc=True)

# localize the datetime objects to UTC and then convert to Belgium time
meteo['datetime'] = meteo['DATEUTC'].apply(lambda x: x.tz_convert('UTC').astimezone(belgium_tz))

meteo.head(3)

meteo['datetime'] = meteo['datetime'].dt.tz_localize(None)
meteo.to_csv('meteo.csv')

file41.groupby(['#object_id', 'description']).size()

print(file41.isna().mean())

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
file41.tail(5)

file41['result_timestamp'] = pd.to_datetime(file41['result_timestamp'])

# extract from timestamp
file41['time'] = file41['result_timestamp'].dt.time
file41['date'] = file41['result_timestamp'].dt.date
file41['hour'] = file41['result_timestamp'].dt.hour
file41['weekday'] = file41['result_timestamp'].dt.strftime('%a')
weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
file41['weekday'] = pd.Categorical(file41['weekday'], categories=weekday_order, ordered=True)
file41.tail(5)

# save the cleaned dataframes to csv files
file40.to_csv('file40.csv')
file41.to_csv('file41.csv')
file42.to_csv('file42.csv')
