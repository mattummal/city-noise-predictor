import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import seaborn as sns
from matplotlib.dates import DateFormatter, MonthLocator

from utils import merge_csv_files

file41 = pd.read_csv('file41.csv')

aggregated_df = (
	file41.groupby(['hour', 'weekday', 'noise_event', 'location']).size().reset_index(name='count')
)
aggregated_df
# For each location
locations = list(aggregated_df.location.unique())
mp01 = aggregated_df[aggregated_df.location == locations[0]].drop(['location'], axis=1)
weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
mp01['weekday'] = pd.Categorical(mp01['weekday'], categories=weekday_order, ordered=True)
mp01

# Heatmap for only Transport sound
mp01_car = mp01[mp01['noise_event'] == 'Transport road - Passenger car'].drop(
	['noise_event'], axis=1
)
mp01_car

# Pivot the data to create a heatmap
heatmap_data = mp01_car.pivot_table(index='hour', columns='weekday', values='count', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', fmt='', cbar=False)
plt.title('Frequency of Transporting sound at MP01')
plt.xlabel('Weekday')
plt.ylabel('Hour')
plt.show()


df_file41 = pd.read_csv(
	's3://teamchadmda/export_41/csv_results_41_255439_mp-01-naamsestraat-35-maxim.csv',
	delimiter=';',
)
df_file41['result_timestamp'] = pd.to_datetime(df_file41['result_timestamp'])
df_file41['date'] = df_file41['result_timestamp'].dt.date
df_file41['hour'] = df_file41['result_timestamp'].dt.hour
df_file41['weekday'] = df_file41['result_timestamp'].dt.strftime('%a')
cols_to_drop = [
	'noise_event_laeq_model_id_unit',
	'noise_event_laeq_model_id',
	'noise_event_laeq_primary_detected_certainty_unit',
	'noise_event_laeq_primary_detected_class_unit',
	'description',
	'#object_id',
	'noise_event_laeq_primary_detected_certainty',
]
df_file41.drop(cols_to_drop, axis=1, inplace=True)
df_file41.rename(columns={'noise_event_laeq_primary_detected_class': 'noise_event'}, inplace=True)
df_file41.head(5)

aggregated_file41 = (
	df_file41.groupby(['hour', 'weekday', 'noise_event']).size().reset_index(name='count')
)
weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
aggregated_file41['weekday'] = pd.Categorical(
	aggregated_file41['weekday'], categories=weekday_order, ordered=True
)
aggregated_file41


sns.set_theme(context='notebook', style='darkgrid')

grouped = file41.groupby(['location', 'noise_event'])['date'].count().reset_index(name='count')

g = sns.catplot(
	data=grouped,
	x='noise_event',
	y='count',
	hue='location',
	kind='bar',
	height=6,
	aspect=1.5,
)

g.set(
	xlabel='Noise Event',
	ylabel='Frequency',
	title='Frequency by Location and Noise Event',
)
plt.xticks(size=8)
plt.tight_layout
plt.show()


# group data by 'hour' and 'noise_event' and count occurrences
grouped = (
	file41.groupby(['hour', 'noise_event'])['noise_event_certainty']
	.count()
	.reset_index(name='count')
)

sns.set_style('darkgrid')
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x='hour', y='count', hue='noise_event', data=grouped)
ax.set_xlabel('Hour')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Noise event by Hour', size=15)
plt.show()


# group data by 'date' and 'noise_event' and count occurrences
grouped = (
	file41.groupby(['date', 'noise_event'])['noise_event_certainty']
	.count()
	.reset_index(name='count')
)

sns.set_style('darkgrid')
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x='date', y='count', hue='noise_event', data=grouped)
ax.set_xlabel('Date')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Noise event by Date', size=15)

# set xticks to show all months
months = MonthLocator()
date_format = DateFormatter('%b')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(date_format)

plt.show()


# plot
grouped = (
	file41.groupby(['weekday', 'noise_event'])['noise_event_certainty']
	.count()
	.reset_index(name='count')
)

sns.set_style('darkgrid')
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x='weekday', y='count', hue='noise_event', data=grouped)
ax.set_xlabel('Weekday')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Noise event by Weekday', size=15)
plt.show()
