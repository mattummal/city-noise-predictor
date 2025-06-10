import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (
	RandomForestRegressor,
	GradientBoostingRegressor,
	ExtraTreesClassifier,
)
from scipy.stats import randint, uniform

weather_data = pd.read_csv('../data/processed_weather_data_leuven.csv')

# Dropping index csv column
weather_data.drop(['Unnamed: 0'], inplace=True, axis=1)

# Format time stamp
weather_data['time'] = pd.to_datetime(weather_data['time'])
weather_data['date'] = weather_data['time'].dt.date
weather_data['hour'] = weather_data['time'].dt.hour
weather_data['month'] = weather_data['time'].dt.month
weather_data['weekday'] = weather_data['time'].dt.strftime('%a')

weather_data = weather_data.groupby(['date', 'hour', 'month', 'weekday']).mean().reset_index()

# Dropping weathercode because signal should be contained in other data + excessive amount of dummies + unseen values
weather_data = weather_data.drop('weathercode', axis=1)

air_quality_data = pd.read_csv('../data/processed_air_quality_data.csv')

# Dropping index csv column
air_quality_data.drop(['Unnamed: 0'], inplace=True, axis=1)

# extract from timestamp
air_quality_data['dt'] = pd.to_datetime(air_quality_data['dt'])
air_quality_data['date'] = air_quality_data['dt'].dt.date
air_quality_data['hour'] = air_quality_data['dt'].dt.hour
air_quality_data['month'] = air_quality_data['dt'].dt.month
air_quality_data['weekday'] = air_quality_data['dt'].dt.strftime('%a')


air_quality_data = (
	air_quality_data.groupby(['date', 'hour', 'month', 'weekday']).mean().reset_index()
)


# Noise data
file40 = pd.read_csv('../data/processed_file40_data.csv')

# Dropping index csv column
file40.drop(['Unnamed: 0'], inplace=True, axis=1)

# Convert the 'result_timestamp' column to a datetime data type
file40['result_timestamp'] = pd.to_datetime(file40['result_timestamp'])
file40['date'] = file40['result_timestamp'].dt.date
file40['hour'] = file40['result_timestamp'].dt.hour
file40['month'] = file40['result_timestamp'].dt.month
file40['weekday'] = file40['result_timestamp'].dt.strftime('%a')

file40 = file40.groupby(['object_id', 'date', 'hour', 'month', 'weekday']).mean().reset_index()

data_model_v2 = file40.merge(
	air_quality_data,
	how='inner',
	left_on=['date', 'hour', 'month', 'weekday'],
	right_on=['date', 'hour', 'month', 'weekday'],
)


data_model_v2 = data_model_v2.merge(
	weather_data,
	how='inner',
	left_on=['date', 'hour', 'month', 'weekday'],
	right_on=['date', 'hour', 'month', 'weekday'],
)


## split train, test data
train_df, val_df = train_test_split(data_model_v2, test_size=0.2, random_state=7)

target_variable = [col for col in train_df.columns if col.startswith('laf')]


y_train = train_df[target_variable]
y_val = val_df[target_variable]

X_train = train_df.drop(target_variable + ['date'], axis=1)
X_val = val_df.drop(target_variable + ['date'], axis=1)

one_hot_var = ['hour', 'month', 'weekday', 'object_id']
numerical_var = [col for col in X_train.columns if col not in one_hot_var]

t = ColumnTransformer(
	transformers=[
		('OneHot', OneHotEncoder(handle_unknown='ignore'), one_hot_var),
		('StandardScaler', StandardScaler(), numerical_var),
	]
)

# fit the encoder
t.fit(X_train, y_train)


# Save encoder
pickle.dump(t, open('../model/model_noise_level_file40/encoder.pkl', 'wb'))


# create pandas DataFrame from dense matrix
X_train = pd.DataFrame(t.fit_transform(X_train), columns=t.get_feature_names_out())
X_val = pd.DataFrame(t.transform(X_val), columns=t.get_feature_names_out())


if os.path.isfile('../model/model_noise_level_file40/laf50_per_hour_dict'):
	model_params = {
		'random_forest': {
			'model': RandomForestRegressor(),
			'params': {
				'n_estimators': randint(50, 100),
				'max_depth': randint(3, 50),
				'max_features': ['auto', 'sqrt'],
				'min_samples_split': randint(2, 20),
				'min_samples_leaf': randint(1, 10),
				'bootstrap': [True, False],
			},
		},
		'gradient_boosting': {
			'model': GradientBoostingRegressor(),
			'params': {
				'n_estimators': randint(50, 100),
				'learning_rate': uniform(0.01, 0.5),
				'max_depth': randint(1, 10),
				'min_samples_split': randint(2, 20),
				'min_samples_leaf': randint(1, 10),
			},
		},
		'xgboost': {
			'model': xgboost.XGBRegressor(),
			'params': {
				'n_estimators': randint(50, 100),
				'learning_rate': uniform(0.01, 0.5),
				'max_depth': randint(1, 10),
				'min_child_weight': randint(1, 10),
				'gamma': uniform(0, 1),
				'reg_alpha': uniform(0, 1),
				'reg_lambda': uniform(0, 1),
			},
		},
	}

# Using RandomizedSearch CV to tune hyperparams, if tuning was done in the past results are loaded from .pckl file
if os.path.isfile('../model/model_noise_level_file40/laf50_per_hour_dict.pkl'):
	print('Params have already been searched and saved, so instead we just load the file')
	params_dict = pickle.load(
		open('../model/model_noise_level_file40/laf50_per_hour_dict.pkl', 'rb')
	)
else:
	# Define the model parameters
	params_dict = {}

	# Loop through each model in model_params and run RandomizedSearchCV
	for model_name, model_info in model_params.items():
		print('Running RandomizedSearchCV for {}...'.format(model_name))

		# Create a RandomizedSearchCV object for the current model
		model = model_info['model']
		param_dist = model_info['params']
		random_search = RandomizedSearchCV(
			model,
			param_distributions=param_dist,
			n_iter=10,
			cv=5,
			n_jobs=1,
			random_state=7,
		)

		# Fit the RandomizedSearchCV object to the data
		random_search.fit(X_train, y_train['laf50_per_hour'])
		params_dict[model_name] = random_search.best_params_

		# Print the best parameters and score
		for model_name, model_info in model_params.items():
			print('Best parameters for {}: '.format(model_name), random_search.best_params_)
			print('Best score for {}: '.format(model_name), random_search.best_score_)
			print('\n')


# Save optimal param dictionary
pickle.dump(params_dict, open('../model/model_noise_level_file40/laf50_per_hour_dict.pkl', 'wb'))

gb_params = params_dict['gradient_boosting']

gb = GradientBoostingRegressor(**gb_params, random_state=7)

gb.fit(X_train, y_train['laf50_per_hour'])

train_preds = gb.predict(X_train)
val_preds = gb.predict(X_val)

print('Train RMSE:', np.sqrt(mean_squared_error(train_preds, y_train['laf50_per_hour'])))
print('Val RMSE:', np.sqrt(mean_squared_error(val_preds, y_val['laf50_per_hour'])))
print('Train MAE:', mean_absolute_error(train_preds, y_train['laf50_per_hour']))
print('Val MAE:', mean_absolute_error(val_preds, y_val['laf50_per_hour']))


rf_params = params_dict['random_forest']

rf = RandomForestRegressor(**rf_params, random_state=7)

rf.fit(X_train, y_train['laf50_per_hour'])

train_preds = rf.predict(X_train)
val_preds = rf.predict(X_val)

print('Train RMSE:', np.sqrt(mean_squared_error(train_preds, y_train['laf50_per_hour'])))
print('Val RMSE:', np.sqrt(mean_squared_error(val_preds, y_val['laf50_per_hour'])))
print('Train MAE:', mean_absolute_error(train_preds, y_train['laf50_per_hour']))
print('Val MAE:', mean_absolute_error(val_preds, y_val['laf50_per_hour']))


xgb_params = params_dict['xgboost']

xgb = xgboost.XGBRegressor(**xgb_params, random_state=7)

xgb.fit(X_train, y_train['laf50_per_hour'])

train_preds = xgb.predict(X_train)
val_preds = xgb.predict(X_val)


print('Train RMSE:', np.sqrt(mean_squared_error(train_preds, y_train['laf50_per_hour'])))
print('Val RMSE:', np.sqrt(mean_squared_error(val_preds, y_val['laf50_per_hour'])))
print('Train MAE:', mean_absolute_error(train_preds, y_train['laf50_per_hour']))
print('Val MAE:', mean_absolute_error(val_preds, y_val['laf50_per_hour']))

plt.scatter(val_preds, y_val['laf50_per_hour'])
plt.xlabel('y pred')
plt.ylabel('y val')


r2_score(val_preds, y_val['laf50_per_hour'])


feature_importances = xgb.feature_importances_
sorted_idx = feature_importances.argsort()[::-1]
sorted_importances = feature_importances[sorted_idx[0:15]]
sorted_columns = list(X_train.columns[sorted_idx[0:15]])
plt.barh(sorted_columns, sorted_importances)


# Saving best model
pickle.dump(xgb, open('../model/model_noise_level_file40/xgb_laf50_per_hour.pkl', 'wb'))


targets = ['laf25_per_hour', 'laf75_per_hour']
model_params_dict = {}
for target in targets:
	if os.path.isfile(f'../model//model_noise_level_file40/{target}_dict.pkl'):
		print('Params have already been searched and saved, so instead we just load the file')
		model_params_dict[target] = pickle.load(
			open(f'../model/model_noise_level_file40/{target}_dict.pkl', 'rb')
		)
	else:
		# Define the model parameters
		model_params = {
			'random_forest': {
				'model': RandomForestRegressor(),
				'params': {
					'n_estimators': randint(50, 100),
					'max_depth': randint(3, 50),
					'max_features': ['auto', 'sqrt'],
					'min_samples_split': randint(2, 20),
					'min_samples_leaf': randint(1, 10),
					'bootstrap': [True, False],
				},
			},
			'gradient_boosting': {
				'model': GradientBoostingRegressor(),
				'params': {
					'n_estimators': randint(50, 100),
					'learning_rate': uniform(0.01, 0.5),
					'max_depth': randint(1, 10),
					'min_samples_split': randint(2, 20),
					'min_samples_leaf': randint(1, 10),
				},
			},
			'xgboost': {
				'model': xgboost.XGBRegressor(),
				'params': {
					'n_estimators': randint(50, 100),
					'learning_rate': uniform(0.01, 0.5),
					'max_depth': randint(1, 10),
					'min_child_weight': randint(1, 10),
					'gamma': uniform(0, 1),
					'reg_alpha': uniform(0, 1),
					'reg_lambda': uniform(0, 1),
				},
			},
		}

		params_dict = {}

		# Loop through each model in model_params and run RandomizedSearchCV
		for model_name, model_info in model_params.items():
			print('Running RandomizedSearchCV for {}...'.format(model_name))

			# Create a RandomizedSearchCV object for the current model
			model = model_info['model']
			param_dist = model_info['params']
			random_search = RandomizedSearchCV(
				model,
				param_distributions=param_dist,
				n_iter=10,
				cv=5,
				n_jobs=1,
				random_state=7,
			)

			# Fit the RandomizedSearchCV object to the data
			random_search.fit(X_train, y_train[target])

			# Print the best parameters and score
			params_dict[model_name] = random_search.best_params_
			print(
				'Best parameters for {}: '.format(model_name),
				random_search.best_params_,
			)
			print('Best score for {}: '.format(model_name), random_search.best_score_)
			print('\n')

		model_params_dict[target] = params_dict
		pickle.dump(params_dict, open(f'../model/model_noise_level_file40/{target}_dict.pkl', 'wb'))


gb_models = {}
for target in targets:
	gb_params = model_params_dict[target]['gradient_boosting']

	gb = GradientBoostingRegressor(**gb_params, random_state=7)

	gb.fit(X_train, y_train[target])

	train_preds = gb.predict(X_train)
	val_preds = gb.predict(X_val)

	print(
		f'Train RMSE of model {target}:',
		np.sqrt(mean_squared_error(train_preds, y_train[target])),
	)
	print(
		f'Val RMSE of model {target}:',
		np.sqrt(mean_squared_error(val_preds, y_val[target])),
	)
	print(
		f'Train MAE of model {target}:',
		mean_absolute_error(train_preds, y_train[target]),
	)
	print(f'Val MAE of model {target}:', mean_absolute_error(val_preds, y_val[target]))
	gb_models[target] = gb


rf_models = {}
for target in targets:
	rf_params = model_params_dict[target]['random_forest']

	rf = RandomForestRegressor(**rf_params, random_state=7)

	rf.fit(X_train, y_train[target])

	train_preds = rf.predict(X_train)
	val_preds = rf.predict(X_val)

	print(
		f'Train RMSE of model {target}:',
		np.sqrt(mean_squared_error(train_preds, y_train[target])),
	)
	print(
		f'Val RMSE of model {target}:',
		np.sqrt(mean_squared_error(val_preds, y_val[target])),
	)
	print(
		f'Train MAE of model {target}:',
		mean_absolute_error(train_preds, y_train[target]),
	)
	print(f'Val MAE of model {target}:', mean_absolute_error(val_preds, y_val[target]))
	rf_models[target] = rf

xgb_models = {}
for target in targets:
	xgb_params = model_params_dict[target]['xgboost']

	xgb = xgboost.XGBRegressor(**xgb_params, random_state=7)
	xgb.fit(X_train, y_train[target])

	train_preds = xgb.predict(X_train)
	val_preds = xgb.predict(X_val)

	print(
		f'Train RMSE of model {target}:',
		np.sqrt(mean_squared_error(train_preds, y_train[target])),
	)
	print(
		f'Val RMSE of model {target}:',
		np.sqrt(mean_squared_error(val_preds, y_val[target])),
	)
	print(
		f'Train MAE of model {target}:',
		mean_absolute_error(train_preds, y_train[target]),
	)
	print(f'Val MAE of model {target}:', mean_absolute_error(val_preds, y_val[target]))
	xgb_models[target] = xgb

# Saving best model (XGB)
for target in targets:
	pickle.dump(
		xgb_models[target],
		open(f'../model/model_noise_level_file40/xgb_{target}.pkl', 'wb'),
	)


# 3 different feature matrices based on the 3 different main data types (time, weather, air_quality)
time_df = data_model_v2[['object_id', 'hour', 'month', 'weekday']]
time_df.name = 'time_df'
weather_df = data_model_v2[
	[
		'object_id',
		'temperature_2m',
		'relativehumidity_2m',
		'dewpoint_2m',
		'apparent_temperature',
		'pressure_msl',
		'surface_pressure',
		'precipitation',
		'rain',
		'snowfall',
		'cloudcover',
		'cloudcover_low',
		'cloudcover_mid',
		'cloudcover_high',
		'shortwave_radiation',
		'direct_radiation',
		'diffuse_radiation',
		'direct_normal_irradiance',
		'windspeed_10m',
		'winddirection_10m',
		'windgusts_10m',
	]
]
weather_df.name = 'weather_df'
air_df = data_model_v2[['object_id', 'pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']]
air_df.name = 'air_df'
feature_dfs = [time_df, air_df, weather_df]

for df in feature_dfs:
	# Using same split random state as before
	df.train, df.val = train_test_split(df, test_size=0.2, random_state=7)


# Building 3 different transformers for the feature matrices

t_time = ColumnTransformer(
	transformers=[('OneHot', OneHotEncoder(handle_unknown='ignore'), time_df.columns)]
)

# fit the encoder
t_time.fit(time_df.train, y_train)
time_df.encoder = t_time

t_air = ColumnTransformer(
	transformers=[
		('OneHot', OneHotEncoder(handle_unknown='ignore'), ['object_id']),
		('StandardScaler', StandardScaler(), ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']),
	]
)

# fit the encoder
t_air.fit(air_df.train, y_train)
air_df.encoder = t_air

t_weather = ColumnTransformer(
	transformers=[
		('OneHot', OneHotEncoder(handle_unknown='ignore'), ['object_id']),
		(
			'StandardScaler',
			StandardScaler(),
			[
				'temperature_2m',
				'relativehumidity_2m',
				'dewpoint_2m',
				'apparent_temperature',
				'pressure_msl',
				'surface_pressure',
				'precipitation',
				'rain',
				'snowfall',
				'cloudcover',
				'cloudcover_low',
				'cloudcover_mid',
				'cloudcover_high',
				'shortwave_radiation',
				'direct_radiation',
				'diffuse_radiation',
				'direct_normal_irradiance',
				'windspeed_10m',
				'winddirection_10m',
				'windgusts_10m',
			],
		),
	]
)

# fit the encoder
t_weather.fit(weather_df.train, y_train)
weather_df.encoder = t_weather


# create pandas DataFrame from dense matrix

time_df.train = pd.DataFrame(
	(time_df.encoder.fit_transform(time_df.train)).toarray(),
	columns=time_df.encoder.get_feature_names_out(),
)
time_df.val = pd.DataFrame(
	(time_df.encoder.fit_transform(time_df.val)).toarray(),
	columns=time_df.encoder.get_feature_names_out(),
)

air_df.train = pd.DataFrame(
	(air_df.encoder.fit_transform(air_df.train)), columns=air_df.encoder.get_feature_names_out()
)
air_df.val = pd.DataFrame(
	(air_df.encoder.fit_transform(air_df.val)), columns=air_df.encoder.get_feature_names_out()
)

weather_df.train = pd.DataFrame(
	(weather_df.encoder.fit_transform(weather_df.train)),
	columns=weather_df.encoder.get_feature_names_out(),
)
weather_df.val = pd.DataFrame(
	(weather_df.encoder.fit_transform(weather_df.val)),
	columns=weather_df.encoder.get_feature_names_out(),
)


model_params_dict = {}

# Checking if Search has been done previously
if os.path.isfile('multiple_features_matrix.pkl'):
	print('Params have already been searched and saved, so instead we just load the file')
	model_params_dict = pickle.load(open('multiple_features_matrix.pkl', 'rb'))
else:
	for df in feature_dfs:
		# Define the model parameters
		model_params = {
			'random_forest': {
				'model': RandomForestRegressor(),
				'params': {
					'n_estimators': randint(50, 100),
					'max_depth': randint(3, 50),
					'max_features': ['auto', 'sqrt'],
					'min_samples_split': randint(2, 20),
					'min_samples_leaf': randint(1, 10),
					'bootstrap': [True, False],
				},
			},
			'gradient_boosting': {
				'model': GradientBoostingRegressor(),
				'params': {
					'n_estimators': randint(50, 100),
					'learning_rate': uniform(0.01, 0.5),
					'max_depth': randint(1, 10),
					'min_samples_split': randint(2, 20),
					'min_samples_leaf': randint(1, 10),
				},
			},
			'xgboost': {
				'model': xgboost.XGBRegressor(),
				'params': {
					'n_estimators': randint(50, 100),
					'learning_rate': uniform(0.01, 0.5),
					'max_depth': randint(1, 10),
					'min_child_weight': randint(1, 10),
					'gamma': uniform(0, 1),
					'reg_alpha': uniform(0, 1),
					'reg_lambda': uniform(0, 1),
				},
			},
		}

		params_dict = {}

		# Loop through each model in model_params and run RandomizedSearchCV
		for model_name, model_info in model_params.items():
			print('Running RandomizedSearchCV for {}...'.format(model_name))

			# Create a RandomizedSearchCV object for the current model
			model = model_info['model']
			param_dist = model_info['params']
			random_search = RandomizedSearchCV(
				model,
				param_distributions=param_dist,
				n_iter=10,
				cv=5,
				n_jobs=1,
				random_state=7,
			)

			# Fit the RandomizedSearchCV object to the data
			random_search.fit(df.train, y_train['laf50_per_hour'])

			# Print the best parameters and score
			params_dict[model_name] = random_search.best_params_
			print(
				'Best parameters for {}: '.format(model_name),
				random_search.best_params_,
			)
			print('Best score for {}: '.format(model_name), random_search.best_score_)
			print('\n')

		model_params_dict[df.name] = params_dict

# Saving the optimal hyper params per model and df
pickle.dump(
	model_params_dict,
	open(f'multiple_features_matrix.pkl', 'wb'),
)


used_models = ['random_forest', 'gradient_boosting', 'xgboost', 'linear_model']
global_model_dictionary = {}
for df in feature_dfs:
	local_model_dictionary = {}
	for model in used_models:
		if model != 'linear_model':
			model_hyper_params = model_params_dict[df.name][model]

		if model == 'gradient_boosting':
			current_model = GradientBoostingRegressor(**model_hyper_params, random_state=7)
		elif model == 'random_forest':
			current_model = RandomForestRegressor(**model_hyper_params, random_state=7)
		elif model == 'linear_model':
			current_model = LinearRegression()
		else:
			current_model = xgboost.XGBRegressor(**model_hyper_params, random_state=7)

		# Fitting and saving model
		current_model.fit(df.train, y_train['laf50_per_hour'])
		local_model_dictionary[model] = current_model

		# Predicting and calculating scores
		train_preds = current_model.predict(df.train)
		val_preds = current_model.predict(df.val)
		test_r2 = r2_score(y_train['laf50_per_hour'], train_preds)
		val_r2 = r2_score(y_val['laf50_per_hour'], val_preds)

		# Saving and printing scores
		local_model_dictionary[model, 'test_r2'] = test_r2
		local_model_dictionary[model, 'val_r2'] = val_r2

		print(f'{df.name}  {model} model test R²: {test_r2}')
		print(f'{df.name}  {model} model validation R²: {val_r2}')
	# Adding all models to dictionary for respective df
	global_model_dictionary[df.name] = local_model_dictionary

# Translating object_ids back into locations for readability
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
location_only = data_model_v2['object_id'].apply(lambda x: object_id_dict[x])
location_train, location_val = train_test_split(location_only, test_size=0.2, random_state=7)


location_train = pd.get_dummies(location_train)
location_val = pd.get_dummies(location_val)

lm = LinearRegression()
lm.fit(location_train, y_train['laf50_per_hour'])
val_pred = lm.predict(location_val)
print(f'R2 coefficen val: {r2_score(y_val["laf50_per_hour"], val_pred)}')


## Lm on full set

lm = LinearRegression()
lm.fit(X_train, y_train['laf50_per_hour'])
val_pred = lm.predict(X_val)
print(f'R2 coefficen val: {r2_score(y_val["laf50_per_hour"], val_pred)}')

## Plotting old TS: the time trend
grouped_data = data_model_v2.groupby(['hour'], as_index=False).mean()
plt.plot(grouped_data['hour'], grouped_data['laf50_per_hour'])
plt.show()


# untrended data
test_df = data_model_v2.copy()
for hour in range(0, 24):
	test_df['laf25_per_hour'] = np.where(
		test_df['hour'] == hour,
		test_df['laf25_per_hour']
		- data_model_v2.loc[data_model_v2['hour'] == hour]['laf25_per_hour'].mean(),
		test_df['laf25_per_hour'],
	)
	test_df['laf50_per_hour'] = np.where(
		test_df['hour'] == hour,
		test_df['laf50_per_hour']
		- data_model_v2.loc[data_model_v2['hour'] == hour]['laf50_per_hour'].mean(),
		test_df['laf50_per_hour'],
	)
	test_df['laf75_per_hour'] = np.where(
		test_df['hour'] == hour,
		test_df['laf75_per_hour']
		- data_model_v2.loc[data_model_v2['hour'] == hour]['laf75_per_hour'].mean(),
		test_df['laf75_per_hour'],
	)

## Plotting New TS
grouped_data = test_df.groupby(['hour'], as_index=False).mean()
plt.plot(grouped_data['hour'], grouped_data['laf50_per_hour'])
plt.show()


## split train, test data
train_df, val_df = train_test_split(test_df, test_size=0.2, random_state=7)

target_variable = ['laf25_per_hour', 'laf50_per_hour', 'laf75_per_hour']

y_train = train_df[target_variable]
y_val = val_df[target_variable]

X_train = train_df.drop(target_variable + ['date'], axis=1)
X_val = val_df.drop(target_variable + ['date'], axis=1)

# create pandas DataFrame from dense matrix
X_train = pd.DataFrame(t.fit_transform(X_train), columns=t.get_feature_names_out())
X_val = pd.DataFrame(t.transform(X_val), columns=t.get_feature_names_out())


model_params_dict = {}

# Checking if Search has been done previously
if os.path.isfile('../model//model_noise_level_file40/detrended_data.pkl'):
	print('Params have already been searched and saved, so instead we just load the file')
	model_params_dict = pickle.load(
		open('../model//model_noise_level_file40/detrended_data.pkl', 'rb')
	)
else:
	for df in feature_dfs:
		# Define the model parameters
		model_params = {
			'xgboost': {
				'model': xgboost.XGBRegressor(),
				'params': {
					'n_estimators': randint(50, 100),
					'learning_rate': uniform(0.01, 0.5),
					'max_depth': randint(1, 10),
					'min_child_weight': randint(1, 10),
					'gamma': uniform(0, 1),
					'reg_alpha': uniform(0, 1),
					'reg_lambda': uniform(0, 1),
				},
			},
		}

		params_dict = {}

		# Loop through each model in model_params and run RandomizedSearchCV
		for model_name, model_info in model_params.items():
			print('Running RandomizedSearchCV for {}...'.format(model_name))

			# Create a RandomizedSearchCV object for the current model
			model = model_info['model']
			param_dist = model_info['params']
			random_search = RandomizedSearchCV(
				model,
				param_distributions=param_dist,
				n_iter=10,
				cv=5,
				n_jobs=1,
				random_state=7,
			)

			# Fit the RandomizedSearchCV object to the data
			random_search.fit(df.train, y_train[target_variable])

			# Print the best parameters and score
			params_dict[model_name] = random_search.best_params_
			print(
				'Best parameters for {}: '.format(model_name),
				random_search.best_params_,
			)
			print('Best score for {}: '.format(model_name), random_search.best_score_)
			print('\n')

		model_params_dict[df.name] = params_dict


# Saving the optimal hyper params per model and df
pickle.dump(
	model_params_dict,
	open(f'../model//model_noise_level_file40/detrended_data.pkl', 'wb'),
)


xgb_params = params_dict['xgboost']

xgb = xgboost.XGBRegressor(**xgb_params, random_state=7)

xgb.fit(X_train, y_train['laf50_per_hour'])

train_preds = xgb.predict(X_train)
val_preds = xgb.predict(X_val)


print('Train RMSE:', np.sqrt(mean_squared_error(train_preds, y_train['laf50_per_hour'])))
print('Val RMSE:', np.sqrt(mean_squared_error(val_preds, y_val['laf50_per_hour'])))
print('Train MAE:', mean_absolute_error(train_preds, y_train['laf50_per_hour']))
print('Val MAE:', mean_absolute_error(val_preds, y_val['laf50_per_hour']))

plt.scatter(val_preds, y_val['laf50_per_hour'])
plt.xlabel('y pred')
plt.ylabel('y val')


r2_score(val_preds, y_val['laf50_per_hour'])

feature_importances = xgb.feature_importances_
sorted_idx = feature_importances.argsort()[::-1]
sorted_importances = feature_importances[sorted_idx[0:15]]
sorted_columns = list(X_train.columns[sorted_idx[0:15]])
plt.barh(sorted_columns, sorted_importances)

data_only_time = data_model_v2.copy()

data_only_time = data_only_time.iloc[:, range(0, 17)]

## split train, test data
train_df, val_df = train_test_split(data_only_time, test_size=0.2, random_state=7)

target_variable = ['laf50_per_hour']

y_train = train_df[target_variable]
y_val = val_df[target_variable]

X_train = train_df.iloc[:, range(0, 5)]
X_val = val_df.iloc[:, range(0, 5)]


t_time = ColumnTransformer(
	transformers=[
		('OneHot', OneHotEncoder(handle_unknown='ignore'), one_hot_var),
	]
)

# fit the encoder
t_time.fit(X_train, y_train)


# create pandas DataFrame from dense matrix
X_train = pd.DataFrame.sparse.from_spmatrix(
	t_time.fit_transform(X_train), columns=t_time.get_feature_names_out()
)
X_val = pd.DataFrame.sparse.from_spmatrix(
	t_time.transform(X_val), columns=t_time.get_feature_names_out()
)


model_params_dict = {}

# Checking if Search has been done previously
if os.path.isfile('../model//model_noise_level_file40/timeonly_data.pkl'):
	print('Params have already been searched and saved, so instead we just load the file')
	model_params_dict = pickle.load(
		open('../model//model_noise_level_file40/timeonly_data.pkl', 'rb')
	)
else:
	model_params = {
		'xgboost': {
			'model': xgboost.XGBRegressor(),
			'params': {
				'n_estimators': randint(50, 100),
				'learning_rate': uniform(0.01, 0.5),
				'max_depth': randint(1, 10),
				'min_child_weight': randint(1, 10),
				'gamma': uniform(0, 1),
				'reg_alpha': uniform(0, 1),
				'reg_lambda': uniform(0, 1),
			},
		},
	}

	params_dict = {}
	# Loop through each model in model_params and run RandomizedSearchCV
	for model_name, model_info in model_params.items():
		print('Running RandomizedSearchCV for {}...'.format(model_name))
		# Create a RandomizedSearchCV object for the current model
		model = model_info['model']
		param_dist = model_info['params']
		random_search = RandomizedSearchCV(
			model,
			param_distributions=param_dist,
			n_iter=10,
			cv=5,
			n_jobs=1,
			random_state=7,
		)

		# Fit the RandomizedSearchCV object to the data
		random_search.fit(X_train, y_train[target_variable])

		# Print the best parameters and score
		params_dict[model_name] = random_search.best_params_
		print(
			'Best parameters for {}: '.format(model_name),
			random_search.best_params_,
		)
		print('Best score for {}: '.format(model_name), random_search.best_score_)
		print('\n')


# Saving the optimal hyper params
pickle.dump(
	params_dict,
	open(f'../model//model_noise_level_file40/timeonly_data.pkl', 'wb'),
)


xgb_params = params_dict['xgboost']

xgb = xgboost.XGBRegressor(**xgb_params, random_state=7)

xgb.fit(X_train, y_train['laf50_per_hour'])

train_preds = xgb.predict(X_train)
val_preds = xgb.predict(X_val)


print('Train RMSE:', np.sqrt(mean_squared_error(train_preds, y_train['laf50_per_hour'])))
print('Val RMSE:', np.sqrt(mean_squared_error(val_preds, y_val['laf50_per_hour'])))
print('Train MAE:', mean_absolute_error(train_preds, y_train['laf50_per_hour']))
print('Val MAE:', mean_absolute_error(val_preds, y_val['laf50_per_hour']))


plt.scatter(val_preds, y_val['laf50_per_hour'])
plt.xlabel('y pred')
plt.ylabel('y val')


r2_score(val_preds, y_val['laf50_per_hour'])


# generate data again
data_only_time = data_model_v2.copy()

data_only_time = data_only_time.iloc[:, range(0, 17)]

## split train, test data
train_df, val_df = train_test_split(data_only_time, test_size=0.2, random_state=7)

target_variable = ['laf50_per_hour']

y_train = train_df[target_variable]
y_val = val_df[target_variable]

X_train = train_df.iloc[:, range(0, 5)]
X_val = val_df.iloc[:, range(0, 5)]

one_hot_var = ['hour', 'month', 'weekday', 'object_id']
t_time = ColumnTransformer(
	transformers=[
		('OneHot', OneHotEncoder(handle_unknown='ignore'), one_hot_var),
	]
)

# fit the encoder
t_time.fit(X_train, y_train)

# create pandas DataFrame from dense matrix
X_train = pd.DataFrame.sparse.from_spmatrix(
	t_time.fit_transform(X_train), columns=t_time.get_feature_names_out()
)
X_val = pd.DataFrame.sparse.from_spmatrix(
	t_time.transform(X_val), columns=t_time.get_feature_names_out()
)


params_dict = pickle.load(open('../model/model_noise_level_file40/timeonly_data.pkl', 'rb'))


R2values = []
xgb_params = params_dict['xgboost'].copy()

n = 5000

for i in range(1000):
	if i % 100 == 0:
		print('Calculating bootstrap sample no. ', i)

	boot_ind = np.random.randint(n, size=n)

	X_train_boot = X_train.iloc[boot_ind]
	y_train_boot = y_train.iloc[boot_ind]
	X_val_boot = X_val.iloc[boot_ind]
	y_val_boot = y_val.iloc[boot_ind]

	xgb = xgboost.XGBRegressor(**xgb_params, random_state=7)
	xgb.fit(X_train_boot, y_train_boot['laf50_per_hour'])

	val_preds_boot = xgb.predict(X_val_boot)

	R2values.append(r2_score(val_preds_boot, y_val_boot['laf50_per_hour']))

R2values = pd.DataFrame(R2values)


print(
	'95% confidence interval for xgboost R2 value: [',
	R2values[0].quantile(0.025),
	', ',
	R2values[0].quantile(0.975),
	']',
)
