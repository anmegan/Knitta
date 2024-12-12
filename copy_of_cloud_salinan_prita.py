# General Purpose
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# Preprocessing and evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# Model building
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU


# Graphs and charts
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
import pickle

url = "Data.cvs"
data = pd.read_csv(url)

print('Number of rows: ', data.shape[0])
print('Number of columns: ', data.shape[1])
print(len(data))

data.head()

data.info() # Print informations about the dataset

# Filter data dari tahun 2018 sampai 2023
filtered_data = data[(data['tahun'] >= 2018) & (data['tahun'] <= 2023)]
print(filtered_data)
print("Sum after filtering: ", len(filtered_data))

# Menghapus lebih dari dua kolom
data.drop(columns=['kode_provinsi', 'nama_provinsi', 'kode_kabupaten_kota'], inplace=True)
print(data)

print('Null Values:', filtered_data.isnull().values.sum())

filtered_data.duplicated().any()
filtered_data[filtered_data.duplicated()]

# Lakukan agregasi untuk mendapatkan total jumlah anak stunting per tahun
summary_per_year = filtered_data.groupby('tahun')['jumlah_balita_stunting'].sum().reset_index()
print(summary_per_year)

#plot tren prior data
plt.figure(figsize=(10, 6))
plt.plot(summary_per_year['tahun'], summary_per_year['jumlah_balita_stunting'], marker='o', color='b', linestyle='-')
plt.title("Tren Jumlah sAnak Stunting (2018-2023)")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Anak Stunting")
plt.grid(True)
plt.xticks(summary_per_year['tahun'].unique())  # Set label sumbu x sesuai tahun yang ada
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.show()

"""LSTM"""

data_values = filtered_data['jumlah_balita_stunting'].values.reshape(-1, 1)

# This cell splits between training data and test data

# Bagi Data: Misalkan 80% untuk training dan 20% untuk testing
training_percentage = 0.80  # Persentase untuk data pelatihan
validation_percentage = 0.20  # Persentase untuk data validasi
training_size = int(len(filtered_data) * training_percentage)
validation_size = int(len(filtered_data) * validation_percentage)
test_size = len(filtered_data) - training_size

# Use .iloc for integer-based location indexing
train_data = filtered_data.iloc[0:training_size, :] # Use .iloc for integer-based indexing
val_data = filtered_data.iloc[training_size:training_size + validation_size, :] # Use .iloc for integer-based indexing
test_data = filtered_data.iloc[training_size + validation_size, :] # Use .iloc for integer-based indexing, select all rows from training_size + validation_size onwards

print("Jumlah data set:", len(filtered_data))
print("Jumlah data train:", len(train_data))
print("Jumlah data validasi:", len(val_data))
print("Jumlah data test:", len(test_data))

print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

# This further splits between x and y

time_step = 30 # This variable declares the time step between every data, trial and error works best

# Training data
X_train, y_train = [], []
for i in range(len(train_data) - time_step - 1):
    a = train_data.iloc[i:(i + time_step), 0]
    X_train.append(a)
    y_train.append(train_data.iloc[i + time_step, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Validation data
X_val, y_val = [], []
for i in range(len(val_data) - time_step - 1):
    a = val_data.iloc[i:(i + time_step), 0]
    X_val.append(a)
    y_val.append(val_data.iloc[i + time_step, 0])

X_val, y_val = np.array(X_val), np.array(y_val)

# Testing data
X_test, y_test = [], []
for i in range(len(test_data) - time_step - 1):
    a = test_data.iloc[i:(i + time_step), 0]
    X_test.append(a)
    y_test.append(test_data.iloc[i + time_step, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_val: ", X_val.shape)  # Print validation data shape
print("y_val: ", y_val.shape)  # Print validation data shape
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# This process adds a dimension using reshape

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)

# Check if X_test has enough data for reshaping
if X_test.size > 0:  # Check if X_test is not empty
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
else:
    print("X_test is empty, skipping reshape.")
    # If X_test is empty you might need to adjust your data splitting
    # or handle this case differently in your model

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

# Normalisasi Data
scaler = MinMaxScaler()
train_values = scaler.fit_transform(train_test)  # Fit dan transform data pelatihan
test_values = scaler.transform(test_values)         # Transform data pengujian menggunakan scaler yang sama
# Cetak hasil
print("Train values after scaling: ", train_values)
print("Test values after scaling: ", test_values)

# Using tensorflow, we can create a LSTM network

model=Sequential()

model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1))) # The LSTM layer
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1))) # The LSTM layer

# You can add as many layer as you want, but remember that quantity is not always the answer
model.add(LSTM(50, return_sequences=False)) # Changed modelLSTM to model

model.add(Dropout(0.1)) # This layer deactivate some neurons from activating

model.add(Dense(30)) # The layer that receives input from all neurons from the previous layer
model.add(Dense(1)) # Like the layer above it, but this one is specialized in outputing the result

model.compile(loss="mean_squared_error", optimizer="adam")

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=5, verbose=1)
# Changed Y_train to y_train and Y_val to y_val to match the variable names defined earlier

# Plots evaluation metrics

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()


plt.show() # Show the plot

# Pastikan X_train dan X_val memiliki bentuk (samples, timesteps, features)
# Misalnya, jika data asli berbentuk (samples, timesteps), kita tambahkan dimensi untuk fitur
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# Lakukan prediksi setelah bentuk data benar
print("LSTM Prediction")
train_predict = model.predict(X_train)  # The train dataset
val_predict = model.predict(X_val)      # The validation/test dataset

train_predict = scaler.inverse_transform(train_predict)
val_predict = scaler.inverse_transform(val_predict)

original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
original_yval = scaler.inverse_transform(y_val.reshape(-1,1))

print("====== LSTM ERROR METRICS ======")
print("VVV TRAIN VVV")
print("Train data RMSE: ", np.sqrt(mean_squared_error(original_ytrain,train_predict))) # The average error squared, go figure
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict)) # The average error squared, go figure
print("Train data MAE: ", mean_absolute_error(original_ytrain,train_predict)) # The average absolute error, go figure
print("VVV VALIDATION VVV")
print("Test data RMSE: ", np.sqrt(mean_squared_error(original_yval,val_predict))) # The average error squared, go figure
print("Test data MSE: ", mean_squared_error(original_yval,val_predict)) # The average error squared, go figure
print("Test data MAE: ", mean_absolute_error(original_yval,val_predict)) # The average absolute error, go figure

from google.colab import drive
drive.mount('/content/drive')



import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_predictions(train_predict, val_predict):
    """
    Plots the training and validation predictions.
    """
    # Assuming original_ytrain, original_yval are defined globally or passed as arguments
    # Adjust according to your data structure

    plt.figure(figsize=(10, 6))
    plt.plot(original_ytrain, color='blue', label='Actual Training Data')
    plt.plot(train_predict, color='red', label='Predicted Training Data')
    plt.plot(original_yval, color='green', label='Actual Validation Data')
    plt.plot(val_predict, color='orange', label='Predicted Validation Data')
    plt.title('Data STunting')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah')
    plt.legend()

# Fit the scaler to your training data (e.g., y_train)
scaler.fit(y_train.reshape(-1, 1))  # Reshape to a 2D array if necessary

# Now you can use the scaler for transform and inverse_transform
train_predict = scaler.inverse_transform(np.array(train_predict).reshape(-1, 1))
val_predict = scaler.inverse_transform(np.array(val_predict).reshape(-1, 1))
original_ytrain = scaler.inverse_transform(np.array(original_ytrain).reshape(-1, 1))  # Assuming original_ytrain was scaled before
original_yval = scaler.inverse_transform(np.array(original_yval).reshape(-1, 1))  # Assuming original_yval was scaled before


# ... (rest of your code) ...
years = np.arange(2010, 2010 + len(original_ytrain))  # Sesuaikan sesuai dengan data pelatihan
years_val = np.arange(2018, 2018 + len(original_yval))  # Sesuaikan sesuai dengan data validasi

# Plot data menggunakan years dan years_val sebagai sumbu x
plt.figure(figsize=(10, 6))
plt.plot(years, original_ytrain, color='blue', label='Actual Training Data')
plt.plot(years, train_predict, color='red', label='Predicted Training Data')
plt.plot(years_val, original_yval, color='green', label='Actual Validation Data')
plt.plot(years_val, val_predict, color='orange', label='Predicted Validation Data')

# Menambahkan judul dan label
plt.title('Data Stunting')
plt.xlabel('Tahun')
plt.ylabel('Jumlah')
plt.legend()
plt.show()

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))