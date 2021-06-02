# Import required modules
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU

# Reading the dataset
dataset_train = 
 pd.read_csv("../input/gooogle-stock-price/Google_Stock_Price_Train.csv")

#Normalization of dataset
training_set = dataset_train.iloc[:,1:2].values
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
plt.plot(training_set_scaled)
plt.title('Google stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.show()

# Divide the dataset into training and validation sets
X_train = []
y_train = []
t = 100 
l = len(training_set_scaled) 
for i in range(t,l):
    X_train.append(training_set_scaled[i-t:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.3)

# Reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

# Creating the model
model = Sequential()
model.add( SimpleRNN (units = 128, return_sequences = True, input_shape = 
                (X_train.shape[1], 1))) 
model.add(LSTM(units = 64, return_sequences = True))
model.add(LSTM(units = 64, return_sequences = True))
model.add(LSTM(units = 32, return_sequences = True))
model.add(GRU(10))
model.add(Dense(units = 1))

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
history=model.fit(X_train, y_train, epochs = 100, batch_size = 32, validation_data = 
            (X_test,y_test))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Testing the model
# Read the data and then perform normalization
dataset_test = pd.read_csv("../input/gooogle-stock-price/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) 
inputs = dataset_total[len(dataset_train)- t: ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# Creating a data structure with 100 timesteps and 1 output
X_test = []
m = len(inputs) 
for i in range(t,m):
    X_test.append(inputs[i-t:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 )) 

# Prediction
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


