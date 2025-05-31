import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pytz
import seaborn as sns
from matplotlib.dates import DateFormatter, MonthLocator

from utils import create_violin_plot, merge_csv_files

file42 = pd.read_csv('file42.csv')

file42.result_timestamp = pd.to_datetime(file42.result_timestamp)
file42['date'] = file42['result_timestamp'].dt.date
file42['month'] = file42['result_timestamp'].dt.month
file42['hour'] = file42['result_timestamp'].dt.hour
file42['weekday'] = file42['result_timestamp'].dt.strftime('%a')

# Make sure weekday and lamax columns are of correct type
file42['weekday'] = pd.Categorical(
	file42['weekday'],
	categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
	ordered=True,
)
file42['lamax'] = file42['lamax'].astype(float)

# Define a color for each day of the week
colors = sns.color_palette('husl', 7)  # 'husl' color palette with 7 colors

# Create a list of traces for each weekday
traces = []
for i, day in enumerate(file42['weekday'].cat.categories):
	color = 'rgb' + str(tuple(int(c * 255) for c in colors[i]))  # Convert color to rgb format
	traces.append(
		go.Violin(x=file42['lamax'][file42['weekday'] == day], line_color=color, name=day)
	)

# Define the layout
layout = go.Layout(
	title='Distribution of Lamax by Weekday',
	xaxis_title='Lamax',
	yaxis_title='Weekday',
	violingap=0,
	violingroupgap=0,
	violinmode='overlay',
)

# Create the figure and add traces
fig = go.Figure(data=traces, layout=layout)

# Show the plot
fig.show()


# Make sure 'hour' and 'lamax' columns are of correct type
file42['hour'] = file42['hour'].astype(int)
file42['lamax'] = file42['lamax'].astype(float)

create_violin_plot(file42, 'hour', 'Distribution of Lamax by Hour')

# Make sure 'month' and 'lamax' columns are of correct type
file42['month'] = file42['month'].astype(int)

create_violin_plot(file42, 'month', 'Distribution of Lamax by Month')


# open files
weather_data = pd.read_csv('../data/processed_weather_data.csv', index_col=0)
air_quality = pd.read_csv('../data/processed_air_quality_data.csv', index_col=0)
file42 = pd.read_csv('../data/processed_file42_data.csv', index_col=0)
# drop NaN
file42.dropna(subset='lamax', inplace=True)
# rename time col
file42.rename(columns={'result_timestamp': 'time'}, inplace=True)
air_quality.rename(columns={'dt': 'time'}, inplace=True)
# merge all df
merged_df = pd.merge(weather_data, air_quality, on=['time', 'hour', 'month'], how='inner')
merged_df = pd.merge(merged_df, file42, on=['time', 'hour', 'month'], how='right')

plt.figure(figsize=(15, 12))
palette = sns.diverging_palette(20, 220, n=256)
corr = merged_df.corr(method='pearson')
sns.heatmap(corr, annot=False, fmt='.2f', cmap=palette, center=0, annot_kws={'size': 8})
plt.title(
	'Correlation Matrix between daily meteorological data and noise measurements',
	size=15,
	weight='bold',
)
plt.show()

# Calculate correlation matrix
corr = merged_df.corr(method='pearson')

# Create heatmap
fig = go.Figure(
	data=go.Heatmap(
		z=corr.values,
		x=corr.columns,
		y=corr.index,
		colorscale='RdBu',
		zmin=-1,
		zmax=1,
		colorbar=dict(title='Correlation'),
	)
)

# Customize layout
fig.update_layout(
	title='Correlation Matrix between daily meteorological data and noise measurements',
	width=900,
	height=900,
	xaxis=dict(title='Columns'),
	yaxis=dict(title='Rows'),
)

# Display the plot
pio.show(fig)
