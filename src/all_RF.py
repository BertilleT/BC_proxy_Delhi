import pandas as pd
from common_fct import remove_nan, print_nan_per, destandardize, BC_plot
from classes import Calibration_Lib
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import time

#path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
path = "../data/preprocessed_delhi_data.csv"
df = pd.read_csv(path)
nb_col = len(df.columns)
#df = df.drop(["PM10"], axis = 1)
#df = remove_nan(df)
#df = df.drop(["date","Hrs."], axis = 1)
#df = df.iloc[:240]
#datetime_df = datetime_df.iloc[:240]
##RANDOM FOREST
train, test = train_test_split(df, test_size=0.25, random_state=42) #by default the data is shuffled
#Standardize
scaler = preprocessing.StandardScaler()
scaler.fit(train)
train_df = pd.DataFrame(scaler.transform(train), columns=train.columns)
test_df = pd.DataFrame(scaler.transform(test), columns=test.columns)

#Training
lib = Calibration_Lib()
X_train = train_df.drop("BC", axis = 1)
Y_train = train_df[["BC"]]
et = time.time()
rf_estimator, best_parameters, train_predicted_Y, RMSE_train, RMSE_validation = lib.train_tune_RF(X_train,Y_train)
st = time.time()
elapsed_time = (et - st) / 60
print('Execution time:', elapsed_time, ' minutes')
print("Best hyper-parameters found during the training: ", best_parameters)
print("RMSE_train: ", RMSE_train)
print("RMSE_validation_cv: ", RMSE_validation)

"""#Validation
validation_df = pd.DataFrame(scaler.transform(validation_df), columns=validation_df.columns)
X_validation = validation_df.drop("BC", axis = 1)
Y_validation = validation_df[["BC"]]
validation_predicted_Y = lib.test_RF(X_validation, Y_validation, best_parameters)
RMSE_validation = np.sqrt(mean_squared_error(Y_validation, validation_predicted_Y))
R2_validation = r2_score(Y_validation, validation_predicted_Y)
print("RMSE_validation: ", RMSE_validation)
print("R2_validation: ", R2_validation)
unscaled_validation_Y, unscaled_validation_predicted_Y = destandardize(Y_validation, validation_predicted_Y, scaler)
unscaled_RMSE_validation = np.sqrt(mean_squared_error(unscaled_validation_Y, unscaled_validation_predicted_Y))
print("unscaled RMSE_validation: ", unscaled_RMSE_validation)"""

#Testing
X_test = test_df.drop("BC", axis = 1)
Y_test = test_df[["BC"]]
test_predicted_Y = lib.test_RF(X_test, Y_test, best_parameters)
RMSE_test = np.sqrt(mean_squared_error(Y_test, test_predicted_Y))
R2_test = r2_score(Y_test, test_predicted_Y)
print("RMSE_test: ", RMSE_test)
print("R2_test: ", R2_test)
unscaled_test_Y, unscaled_test_predicted_Y = destandardize(Y_test, test_predicted_Y, scaler, nb_col)
unscaled_RMSE_test = np.sqrt(mean_squared_error(unscaled_test_Y, unscaled_test_predicted_Y))
print("unscaled RMSE_test: ", unscaled_RMSE_test)

#BC_plot(unscaled_train_Y, unscaled_train_predicted_Y, datetime_df, scaler, "train")
#BC_plot(unscaled_test_Y, unscaled_test_predicted_Y, datetime_df, scaler, "test")