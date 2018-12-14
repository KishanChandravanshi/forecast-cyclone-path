# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model

############################## Predict X-Coordinate##########################
filename = 'final.csv' # name of your csv file
timestep = 5 # how many coordinate to consider before predicting
end_term = 120 # total coordinates used for training
epochs = 1000
# Importing the training set
dataset_train = pd.read_csv(filename)
training_set = dataset_train.iloc[:end_term, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc1 = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc1.fit_transform(training_set)

# Creating a data structure with 20 timesteps and 1 output
X_train = []
y_train = []
for i in range(timestep, end_term):
    X_train.append(training_set_scaled[i-timestep:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


################################# TRAINING FOR X #############################
# Initialising the RNN
regressorx = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressorx.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressorx.add(Dropout(0.5))

# Adding a second LSTM layer and some Dropout regularisation
regressorx.add(LSTM(units = 100, return_sequences = True))
regressorx.add(Dropout(0.5))

# Adding a third LSTM layer and some Dropout regularisation
regressorx.add(LSTM(units = 100, return_sequences = True))
regressorx.add(Dropout(0.5))

regressorx.add(LSTM(units = 100, return_sequences = True))
regressorx.add(Dropout(0.5))

regressorx.add(LSTM(units = 100, return_sequences = True))
regressorx.add(Dropout(0.5))

# Adding a fourth LSTM layer and some Dropout regularisation
regressorx.add(LSTM(units = 100))
regressorx.add(Dropout(0.5))

# Adding the output layer
regressorx.add(Dense(units = 1))

# Compiling the RNN
regressorx.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressorx.fit(X_train, y_train, epochs = epochs, batch_size = 16)

regressorx.save('x_weight_5_1000.h5')
# Part 3 - Making the predictions and visualising the results
# predicting the midpoint trend afterwards
dataset_test = pd.read_csv(filename)
real_midpoint_x = dataset_test.iloc[end_term:, 1:2].values
inputs = dataset_test.iloc[end_term - timestep:, 1:2].values
inputs = inputs.reshape(-1,1)
inputs = sc1.transform(inputs)
# Creating the test data format compatible to be fed
X_test = []
for i in range(timestep, len(inputs)):
    X_test.append(inputs[i-timestep:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_midpoint_x = regressorx.predict(X_test)
predicted_midpoint_x = sc1.inverse_transform(predicted_midpoint_x)

# Visualising the results
plt.plot(real_midpoint_x, color = 'red', label = 'Original X - Coordinate')
plt.plot(predicted_midpoint_x, color = 'blue', label = 'Predicted X - Coordinate')
plt.title('X - Coordinate Prediction')
plt.xlabel('Time')
plt.ylabel('X - Coordinate')
plt.legend()
plt.show()




################################# TRAINING FOR Y #############################
# Importing the training set
dataset_train = pd.read_csv(filename)
training_set = dataset_train.iloc[:end_term, 2:3].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc2 = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc2.fit_transform(training_set)

# Creating a data structure with 5 timesteps and 1 output
X_train = []
y_train = []
for i in range(timestep, end_term):
    X_train.append(training_set_scaled[i-timestep:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Initialising the RNN
regressory = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressory.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressory.add(Dropout(0.5))

# Adding a second LSTM layer and some Dropout regularisation
regressory.add(LSTM(units = 100, return_sequences = True))
regressory.add(Dropout(0.5))

# Adding a third LSTM layer and some Dropout regularisation
regressory.add(LSTM(units = 100, return_sequences = True))
regressory.add(Dropout(0.5))

# Adding a fourth LSTM layer and some Dropout regularisation
regressory.add(LSTM(units = 100, return_sequences = True))
regressory.add(Dropout(0.5))

# Adding a fifth LSTM layer and some Dropout regularisation
regressory.add(LSTM(units = 100, return_sequences = True))
regressory.add(Dropout(0.5))

# Adding the output layer
regressory.add(Dense(units = 1))

# Compiling the RNN
regressory.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressory.fit(X_train, y_train, epochs = epochs, batch_size = 16)
regressory.save('y_weight_5_1000.h5')



# Part 3 - Making the predictions and visualising the results

dataset_test = pd.read_csv(filename)
real_midpoint_y = dataset_test.iloc[end_term:, 2:3].values
inputs = dataset_test.iloc[end_term - timestep:, 2:3].values
inputs = inputs.reshape(-1,1)
inputs = sc2.transform(inputs)

X_test = []
for i in range(timestep, len(inputs)):
    X_test.append(inputs[i-timestep:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_midpoint_y = regressory.predict(X_test)
predicted_midpoint_y = sc2.inverse_transform(predicted_midpoint_y)

# Visualising the results
plt.plot(real_midpoint_y, color = 'red', label = 'Original Y - Coordinate')
plt.plot(predicted_midpoint_y, color = 'blue', label = 'Predicted Y - Coordinate')
plt.title('Y - Coordinate Prediction')
plt.xlabel('Time')
plt.ylabel('Y - Coordinate')
plt.legend()
plt.show()

# # Final Ouput with bounding box across the image for single image

# load the model
regressorx = load_model('x_weight_5_1000.h5')
regressory = load_model('y_weight_5_1000.h5')
filename = 'final.csv'

import cv2
dataset = pd.read_csv(filename)
img = cv2.imread('last_sequence.jpg')

x_input = np.array(dataset.iloc[145 - timestep: 145, 1:2])
x_input = x_input.reshape(1, timestep)
x_input = sc1.transform(x_input)
x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

y_input = np.array(dataset.iloc[145 - timestep: 145, 2:3])
y_input = y_input.reshape(1, timestep)
y_input = sc2.transform(y_input)
y_input = np.reshape(y_input, (y_input.shape[0], y_input.shape[1], 1))

x_predicted = sc1.inverse_transform(regressorx.predict(x_input))
y_predicted = sc2.inverse_transform(regressory.predict(y_input))

# change the radius to any desired value
img_new = cv2.circle(img, center=(x_predicted, y_predicted), radius=20, color=(0, 0, 255), thickness=3)
cv2.imshow('image', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
