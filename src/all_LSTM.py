"authors: Iva Bokšić and Bertille Temple"

from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from common_fct import remove_nan, print_nan_per, destandardize, split, filter_df, sample_split, BC_plot
from sklearn.model_selection import train_test_split


path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
df = pd.read_excel(path)
df = remove_nan(df)
df = df.drop(["date","Hrs."], axis = 1)
# Separate the features and target variable
X = df.drop('BC', axis=1)  # Features
y = df['BC']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale the data
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

y_train=y_train_scaled
y_test=y_test_scaled

# Reshape the input data for LSTM
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
# Build the LSTM model
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
model.add(Dense(1))  # Output layer with 1 neuron for regression

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, batch_size=32, epochs=50, validation_data=(X_test_reshaped, y_test))

# Predict on train and test sets
y_train_pred = model.predict(X_train_reshaped).flatten()
y_test_pred = model.predict(X_test_reshaped).flatten()

# Calculate MSE for train and test predictions
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss (LSTM)')
plt.legend()
plt.show()

# Calculate RMSE for train and test predictions
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Train RMSE: {:.4f}".format(train_rmse))
print("Test RMSE: {:.4f}".format(test_rmse))

# Calculate R2 score for train and test predictions
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Train R2 Score: {:.4f}".format(train_r2))
print("Test R2 Score: {:.4f}".format(test_r2))


unscaled_test_Y, unscaled_test_predicted_Y = destandardize(Y_test, test_predicted_Y, scaler, nb_col)
unscaled_RMSE_test = np.sqrt(mean_squared_error(unscaled_test_Y, unscaled_test_predicted_Y))
print("unscaled RMSE_test: ", unscaled_RMSE_test)