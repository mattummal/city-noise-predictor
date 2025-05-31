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

file40 = pd.read_csv('file40.csv')

# drop all _unit columns
cols_to_drop = [col for col in file40.columns if col.endswith('unit')]
file40.drop(cols_to_drop, axis=1, inplace=True)

# rename columns
file40.rename(columns={'description': 'location'}, inplace=True)


# Convert the 'result_timestamp' column to a datetime data type
file40['result_timestamp'] = pd.to_datetime(file40['result_timestamp'])
file40['date'] = file40['result_timestamp'].dt.date
file40['hour'] = file40['result_timestamp'].dt.hour
file40['weekday'] = file40['result_timestamp'].dt.strftime('%a')


laf_cols = [col for col in file40.columns if col.startswith('laf')]


locations = file40['location'].unique()

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 12), sharey=True)

# loop over all location values and plot them in the grid
for i, loc in enumerate(locations):
	row = i // 4
	col = i % 4

	# filter the data for the current location and group by hour
	loc_data = file40[file40['location'] == loc]
	loc_hour = loc_data.groupby('hour')[laf_cols].mean()

	# create the line plot for each LAF column
	for var in laf_cols:
		sns.lineplot(data=loc_hour[var], label=None, ax=axs[row, col])

	axs[row, col].set_title(f'{loc}')
	axs[row, col].set_xlabel('Hour')
	axs[row, col].set_ylabel('dB(A)')
	axs[row, col].set_xticks(loc_hour.index)

legend = fig.legend(laf_cols, title='LAF values', loc='lower right', bbox_to_anchor=(1.1, 0.5))

# add a title to the whole plot
fig.suptitle('Mean LAF by hour and location')

plt.tight_layout()
plt.show()


file40[file40.laf005_per_hour > 100]


locations = file40['location'].unique()

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 12), sharey=True)

# loop over all location values and plot them in the grid
for i, loc in enumerate(locations):
	row = i // 4
	col = i % 4

	# filter the data for the current location and group by hour
	loc_data = file40[file40['location'] == loc]
	loc_date = loc_data.groupby('date')[laf_cols].mean()

	# create the line plot for each LAF column
	for var in laf_cols:
		sns.lineplot(data=loc_date[var], label=None, ax=axs[row, col])

	axs[row, col].set_title(f'{loc}')
	axs[row, col].set_xlabel('Date')
	axs[row, col].set_ylabel('dB(A)')

legend = fig.legend(laf_cols, title='LAF values', loc='lower right', bbox_to_anchor=(1.1, 0.5))

# add a title to the whole plot
fig.suptitle('Mean LAF by date and location')

plt.tight_layout()
plt.show()


locations = file40['location'].unique()

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 12), sharey=True)

# loop over all location values and plot them in the grid
for i, loc in enumerate(locations):
	row = i // 4
	col = i % 4

	# filter the data for the current location and group by hour
	loc_data = file40[file40['location'] == loc]
	loc_date = loc_data.groupby('date')['laf05_per_hour', 'laf50_per_hour', 'laf95_per_hour'].max()

	# create the line plot for each LAF column
	for var in ['laf05_per_hour', 'laf50_per_hour', 'laf95_per_hour']:
		sns.lineplot(data=loc_date[var], label=None, ax=axs[row, col])

	axs[row, col].set_title(f'{loc}')
	axs[row, col].set_xlabel('Date')
	axs[row, col].set_ylabel('dB(A)')

legend = fig.legend(
	['laf05_per_hour', 'laf50_per_hour', 'laf95_per_hour'],
	title='LAF values',
	loc='lower right',
	bbox_to_anchor=(1.1, 0.5),
)

# add a title to the whole plot
fig.suptitle('Max Laf values by date and location')

plt.tight_layout()
plt.show()


locations = file40['location'].unique()

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 12), sharey=True)

# loop over all location values and plot them in the grid
for i, loc in enumerate(locations):
	row = i // 4
	col = i % 4

	# filter the data for the current location and group by hour
	loc_data = file40[file40['location'] == loc]
	loc_date = loc_data.groupby('weekday')[
		'laf05_per_hour', 'laf50_per_hour', 'laf95_per_hour'
	].mean()

	# create the line plot for each LAF column
	for var in ['laf05_per_hour', 'laf50_per_hour', 'laf95_per_hour']:
		sns.lineplot(data=loc_date[var], label=None, ax=axs[row, col])

	axs[row, col].set_title(f'{loc}')
	axs[row, col].set_xlabel('Weekday')
	axs[row, col].set_ylabel('dB(A)')

legend = fig.legend(
	['laf05_per_hour', 'laf50_per_hour', 'laf95_per_hour'],
	title='LAF values',
	loc='lower right',
	bbox_to_anchor=(1.1, 0.5),
)

# add a title to the whole plot
fig.suptitle('Mean Laf values by weekday and location')

plt.tight_layout()
plt.show()


# Resample noise level by day in all locations
laf_cols = [col for col in file40.columns if col.startswith('laf')]
noise_level_daily_mean = file40.copy()
noise_level_daily_mean.rename(columns={'result_timestamp': 'datetime'}, inplace=True)
noise_level_daily_mean.set_index('datetime', inplace=True)
noise_level_daily_mean = noise_level_daily_mean[laf_cols].resample('D').mean()

sns.lineplot(data=noise_level_daily_mean)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='LAF values', frameon=False)
plt.title('Mean LAF by date in all locations')
plt.show()


# set time as index
meteo = pd.read_csv('meteo.csv')
# show max columns
pd.set_option('display.max_columns', None)
meteo.set_index('datetime', inplace=True)
# resample the data by day and take the mean
lc_cols = [col for col in meteo.columns if col.startswith('LC')]
meteo_daily_mean = meteo[lc_cols].resample('D').mean()
meteo_hourly_mean = meteo[lc_cols].resample('H').mean()
meteo_daily_mean.reset_index(inplace=True)
meteo_hourly_mean.reset_index(inplace=True)
meteo_hourly_mean.to_csv('meteo_hourly.csv')


for var in lc_cols:
	plt.figure()
	sns.lineplot(data=meteo_daily_mean, x='datetime', y=meteo_daily_mean[var])
	plt.title(col)
	plt.xlabel('Date')
	plt.ylabel(var)


# Merge meteo daily data with mean daily noise level data
noise_level_daily_mean.reset_index(inplace=True)
meteo_noise_daily_mean = meteo_daily_mean.merge(
	noise_level_daily_mean[['datetime', 'laf50_per_hour']], on=['datetime']
)
meteo_noise_daily_mean.head(3)


plt.figure(figsize=(15, 12))
palette = sns.diverging_palette(20, 220, n=256)
corr = meteo_noise_daily_mean.corr(method='pearson')
sns.heatmap(corr, annot=True, fmt='.2f', cmap=palette, center=0, annot_kws={'size': 8})
plt.title(
	'Correlation Matrix between daily meteorological data and LAF value',
	size=15,
	weight='bold',
)
plt.show()

# resample the data by hour and take the mean
lc_cols = [col for col in meteo.columns if col.startswith('LC')]
meteo_hourly_mean = meteo[lc_cols].resample('H').mean()
meteo_hourly_mean.reset_index(inplace=True)

# Resample noise level by day in all locations
laf_cols = [col for col in file40.columns if col.startswith('laf')]
noise_level_hourly_mean = file40.copy()
noise_level_hourly_mean.rename(columns={'result_timestamp': 'datetime'}, inplace=True)
noise_level_hourly_mean.set_index('datetime', inplace=True)
noise_level_hourly_mean = noise_level_hourly_mean[laf_cols].resample('H').mean()
noise_level_hourly_mean.to_csv('file40_hourly.csv')

# Merge meteo daily data with mean daily noise level data
noise_level_hourly_mean.reset_index(inplace=True)
meteo_noise_hourly_mean = meteo_hourly_mean.merge(
	noise_level_hourly_mean[['datetime', 'laf50_per_hour']], on=['datetime']
)

plt.figure(figsize=(15, 12))
palette = sns.diverging_palette(20, 220, n=256)
corr = meteo_noise_hourly_mean.corr(method='pearson')
sns.heatmap(corr, annot=True, fmt='.2f', cmap=palette, center=0, annot_kws={'size': 8})
plt.title(
	'Correlation Matrix between hourly meteorological data and LAF value',
	size=15,
	weight='bold',
)
plt.show()
