import pandas as pd
from common_fct import remove_nan, destandardize
from classes import Calibration_Lib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
import time

"""path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
df = pd.read_excel(path)
df = df.drop(["PM10"], axis = 1)
df = remove_nan(df)
df['datetime'] = df['date'].astype(str) + ' ' + df['Hrs.'].astype(str)
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M:%S')
datetime_df = df[["datetime"]].reset_index(drop=True)
df = df.drop(["date","Hrs.", "datetime"], axis = 1)"""
#df = df.iloc[:240]
#datetime_df = datetime_df.iloc[:240]
path = "../data/preprocessed_delhi_data.csv"
df = pd.read_csv(path)
nb_col = len(df.columns)
#SVR
train, test = train_test_split(df, test_size=0.25, random_state=42)
#standardize
scaler = preprocessing.StandardScaler()
scaler.fit(train)
train_df = pd.DataFrame(scaler.transform(train), columns=train.columns)
test_df = pd.DataFrame(scaler.transform(test), columns=test.columns)

#Training
lib = Calibration_Lib()
X_train = train_df.drop("BC", axis = 1)
Y_train = train_df[["BC"]]
st = time.time()
svr_estimator, best_parameters, train_predicted_Y, RMSE_train, RMSE_validation = lib.train_tune_SVR(X_train,Y_train)
et = time.time()
elapsed_time = (et - st) / 60
print('Execution time:', elapsed_time, ' minutes')
print("Best hyper-parameters found during the training: ", best_parameters)
print("RMSE_train: ", RMSE_train)
print("RMSE_validation_cv: ", RMSE_validation)

#Testing
X_test = test_df.drop("BC", axis = 1)
Y_test = test_df[["BC"]]
test_predicted_Y = lib.test_SVR(X_test, Y_test, best_parameters)
RMSE_test = np.sqrt(mean_squared_error(Y_test, test_predicted_Y))
R2_test = r2_score(Y_test, test_predicted_Y)
print("RMSE_test: ", RMSE_test)
print("R2_test: ", R2_test)

#unscaled_train_Y, unscaled_train_predicted_Y = destandardize(Y_train, train_predicted_Y, scaler, nb_col)
unscaled_test_Y, unscaled_test_predicted_Y = destandardize(Y_test, test_predicted_Y, scaler, nb_col)
unscaled_RMSE_test = np.sqrt(mean_squared_error(unscaled_test_Y, unscaled_test_predicted_Y))
print("unscaled RMSE_test: ", unscaled_RMSE_test)