# univariate onestep mlp example
from math import sqrt
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# MLP
def mlp(X, y, n_steps):
	# train and test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	# define model
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=n_steps))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	# fit model
	model.fit(X_train, y_train, epochs=10, verbose=0)
	# demonstrate prediction
	yhat = model.predict(X_test, verbose=0)
	print("MLP RMSE:", measure_rmse(y_test, yhat))

# define dataset
series = read_csv('household_power_consumption_months.csv', header=0, index_col=0, usecols=['datetime', 'Global_active_power'])
data = series.values
# choose a number of time steps
n_steps = 8
# split into samples
X, y = split_sequence(data, n_steps)

# call functions for train and test models
mlp(X, y, n_steps)


