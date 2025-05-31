import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import os
import plotly.express as px
import plotly.graph_objects as go
import requests


def merge_csv_files(directory_path):
	"""
	Merges all CSV files in a directory into a single pandas DataFrame.

	Args:
	directory_path (str): The directory path containing the CSV files.

	Returns:
	merged_df (pandas.DataFrame): The merged pandas DataFrame of all CSV files in the directory.
	"""
	dfs = []

	# loop through each file in the directory
	for file in os.listdir(directory_path):
		# check if the file is a CSV file
		if file.endswith('.csv'):
			file_path = os.path.join(directory_path, file)
			if directory_path.endswith('dataverse_file'):
				df = pd.read_csv(file_path, delimiter=',')
			else:
				df = pd.read_csv(file_path, delimiter=';')
			dfs.append(df)

	# concatenate all dataframes
	merged_df = pd.concat(dfs, ignore_index=True)

	return merged_df


def add_object_id(hourly_df, object_ids):
	dfs = []
	for id in object_ids:
		new_df = hourly_df.copy()
		new_df['object_id'] = id
		dfs.append(new_df)

	combined_df = pd.concat(dfs)
	combined_df.reset_index(drop=True, inplace=True)
	return combined_df


def create_violin_plot(df, groupby_col, title):
	# Define a color palette
	num_unique_values = df[groupby_col].nunique()
	colors = sns.color_palette('husl', num_unique_values)

	# Create a list of traces for each unique value in the groupby column
	traces = []
	for i, val in enumerate(sorted(df[groupby_col].unique())):
		color = 'rgb' + str(tuple(int(c * 255) for c in colors[i]))  # Convert color to rgb format
		traces.append(
			go.Violin(x=df['lamax'][df[groupby_col] == val], line_color=color, name=str(val))
		)

	# Define the layout
	layout = go.Layout(
		title=title,
		xaxis_title='Lamax',
		yaxis_title=groupby_col.capitalize(),
		violingap=0,
		violingroupgap=0,
		violinmode='overlay',
	)

	# Create the figure and add traces
	fig = go.Figure(data=traces, layout=layout)

	# Show the plot
	fig.show()


def get_forecast_hourly_weather(url):
	resp = requests.get(url)
	data = resp.json()
	df = pd.DataFrame(data['hourly'])
	return df


def process_file(file_path):
	print(f'Processing: {file_path}')
	df = pd.read_csv(file_path, delimiter=';')

	# convert 'result_timestamp' to datetime format
	df['result_timestamp'] = pd.to_datetime(df['result_timestamp'], format='%d/%m/%Y %H:%M:%S.%f')

	# set 'result_timestamp' as the index
	df.set_index('result_timestamp', inplace=True)

	# resample to hourly frequency and calculate the mean
	df_resampled = df.resample('H').mean()

	# reset index and add additional columns
	df_resampled.reset_index(inplace=True)
	df_resampled['date'] = df_resampled['result_timestamp'].dt.date
	df_resampled['hour'] = df_resampled['result_timestamp'].dt.hour
	df_resampled['weekday'] = df_resampled['result_timestamp'].dt.strftime('%a')
	df_resampled['month'] = df_resampled['result_timestamp'].dt.month

	# drop rows with NaN values in 'lamax' column
	df_resampled.dropna(subset=['lamax'], inplace=True)

	# rename column and convert 'object_id' to int
	df_resampled.rename(columns={'#object_id': 'object_id'}, inplace=True)
	df_resampled['object_id'] = df_resampled['object_id'].astype(int)

	# ddd location column using 'object_id_dict'
	df_resampled['location'] = df_resampled['object_id'].map(object_id_dict)
	return df_resampled
