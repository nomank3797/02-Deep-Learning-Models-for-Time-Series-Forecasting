# multivariate multiple input series multistep lstm example
from math import sqrt
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, 0]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	scores = list()
	# calculate an RMSE score for each series
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# Vanilla LSTM
def vanilla_lstm(X, y, n_steps_in, n_steps_out):
	n_features = X.shape[2]
	# train and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	# define model
	model = Sequential()
	model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X_train, y_train, epochs=10, verbose=0)
	# demonstrate prediction
	yhat = model.predict(X_test, verbose=0)
	print("Vanilla LSTM RMSE:", measure_rmse(y_test, yhat))

# Stacked LSTM	
def stacked_lstm(X, y, n_steps_in, n_steps_out):
	n_features = X.shape[2]
	X = X.reshape((X.shape[0], X.shape[1], n_features))
	# train and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	# define model
	model = Sequential()
	model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
	model.add(LSTM(50, activation='relu'))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X_train, y_train, epochs=10, verbose=0)
	# demonstrate prediction
	yhat = model.predict(X_test, verbose=0)
	print("Stacked LSTM RMSE:", measure_rmse(y_test, yhat))

# Bidirectional LSTM
def blstm(X, y, n_steps_in, n_steps_out):
	n_features = X.shape[2]
	X = X.reshape((X.shape[0], X.shape[1], n_features))
	# train and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	# define model
	model = Sequential()
	model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps_in, n_features)))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X_train, y_train, epochs=10, verbose=0)
	# demonstrate prediction
	yhat = model.predict(X_test, verbose=0)
	print("Bidirectional LSTM RMSE:", measure_rmse(y_test, yhat))

# CNN-LSTM
def cnn_lstm(X, y, n_steps_in, n_steps_out):
	# reshape from [samples, timesteps, features] into [samples, subsequences, timesteps, features]
	n_features = X.shape[2]
	n_seq = 2
	n_steps = 4
	X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
	# train and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	# define model
	model = Sequential()
	model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, padding="SAME", activation='relu'), input_shape=(None, n_steps, n_features)))
	model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	model.add(TimeDistributed(Flatten()))
	model.add(LSTM(50, activation='relu'))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X_train, y_train, epochs=10, verbose=0)
	# demonstrate prediction
	yhat = model.predict(X_test, verbose=0)
	print("CNN-LSTM RMSE:", measure_rmse(y_test, yhat))

# ConvLSTM2D
def convlstm2D(X, y, n_steps_in, n_steps_out):
	# reshape from [samples, timesteps, features] into [samples, timesteps, rows, columns, features]
	n_features = X.shape[2]
	n_seq = 2
	n_steps = 4
	X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
	# train and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	# define model
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
	model.add(Flatten())
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X_train, y_train, epochs=10, verbose=0)
	# demonstrate prediction
	yhat = model.predict(X_test, verbose=0)
	print("ConvLSTM2D RMSE:", measure_rmse(y_test, yhat))

# define dataset
data = read_csv('household_power_consumption_months.csv', header=0, index_col=0)
data = data.values
# choose a number of time steps
n_steps_in, n_steps_out = 8, 8
# convert into input/output
X, y = split_sequences(data, n_steps_in, n_steps_out)
print(X.shape,y.shape)

# call functions for train and test models
vanilla_lstm(X, y, n_steps_in, n_steps_out)
stacked_lstm(X, y, n_steps_in, n_steps_out)
blstm(X, y, n_steps_in, n_steps_out)
cnn_lstm(X, y, n_steps_in, n_steps_out)
convlstm2D(X, y, n_steps_in, n_steps_out)





