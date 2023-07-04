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
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
# Now since we want the valid and test size to be equal (10% each of overall data). 
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)
X_test_scaled = scaler.transform(X_test)

scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_valid_scaled = scaler.fit_transform(y_valid.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

y_train = y_train_scaled
y_valid = y_valid_scaled
y_test = y_test_scaled

# Reshape the input data for LSTM
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_valid_reshaped = np.reshape(X_valid_scaled, (X_valid_scaled.shape[0], 1, X_valid_scaled.shape[1]))
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
y_valid_pred = model.predict(X_valid_reshaped).flatten()
y_test_pred = model.predict(X_test_reshaped).flatten()

# Calculate RMSE for train and test predictions
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
valid_mae = mean_absolute_error(y_valid, y_valid_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

# Calculate R2 score for train and test predictions
train_r2 = r2_score(y_train, y_train_pred)
valid_r2 = r2_score(y_valid, y_valid_pred)
test_r2 = r2_score(y_test, y_test_pred)

unscaled_valid_Y, unscaled_valid_predicted_Y = destandardize(y_valid, y_valid_pred, scaler, 6)
unscaled_RMSE_valid = np.sqrt(mean_squared_error(unscaled_valid_Y, unscaled_valid_predicted_Y))
unscaled_MAE_valid = mean_absolute_error(unscaled_valid_Y, unscaled_valid_predicted_Y)

unscaled_test_Y, unscaled_test_predicted_Y = destandardize(y_test, y_test_pred, scaler, 6)
unscaled_RMSE_test = np.sqrt(mean_squared_error(unscaled_test_Y, unscaled_test_predicted_Y))
unscaled_MAE_test = mean_absolute_error(unscaled_test_Y, unscaled_test_predicted_Y)

print("Train RMSE: {:.4f}".format(train_rmse))
print("Validation RMSE: {:.4f}".format(valid_rmse))
print("unscaled RMSE_validation: ", unscaled_RMSE_valid)
#print("Test RMSE: {:.4f}".format(test_rmse))
#print("unscaled RMSE_test: ", unscaled_RMSE_test)
print("Train MAE: {:.4f}".format(train_mae))
print("Validation MAE: {:.4f}".format(valid_mae))
print("unscaled MAE_validation: ", unscaled_MAE_valid)
#print("Test MAE: {:.4f}".format(test_mae))

print("Train R2 Score: {:.4f}".format(train_r2))
print("Validation R2 Score: {:.4f}".format(valid_r2))
#print("Test R2 Score: {:.4f}".format(test_r2))
#print("unscaled MAE_test: ", unscaled_MAE_test)

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss (LSTM)')
plt.legend()
plt.show()