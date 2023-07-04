import pandas as pd
from common_fct import remove_nan, print_nan_per, destandardize, split, sample_split, BC_plot
from classes import BC_proxy_lib
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime
import time 
import matplotlib.pyplot as plt

path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
df = pd.read_excel(path)

##SVR
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
winter_df, pre_monsoon_df, summer_df, post_monsoon_df = split(df)

winter_sample_df = sample_split("2018-12-16", "2018-12-20", df)
pre_monsoon_sample_df = sample_split("2019-03-18", "2019-03-24", df)
summer_sample_df = sample_split("2019-06-20", "2019-06-24", df)
post_monsoon_sample_df = sample_split("2019-11-01", "2019-11-04", df)

seasons_df = {"winter": winter_df, "pre_monsoon": pre_monsoon_df, "summer": summer_df, "post_monsoon": post_monsoon_df}
sample_seasons_df = {"winter": winter_sample_df, "pre_monsoon": pre_monsoon_sample_df, "summer": summer_sample_df, "post_monsoon": post_monsoon_sample_df}
keys_list = list(seasons_df.keys())
season = keys_list[3]
df = seasons_df[season]
print("--------------------------------------------------------------------------------------------------------------------------------------")
df = df.drop(["date","Hrs."], axis = 1)
df = remove_nan(df)
df = df.sample(frac=1, random_state=42) #shuffle
nb_col = len(df.columns)
print("Season under study: ", season)
train, test = train_test_split(df, test_size=0.25, random_state=42) #by default the data is shuffled
train = df
#Standardize
scaler = preprocessing.StandardScaler()
scaler.fit(train)
train_df = pd.DataFrame(scaler.transform(train), columns=train.columns)
test_df = pd.DataFrame(scaler.transform(test), columns=test.columns)

#Training
lib = BC_proxy_lib()
X_train = train_df.drop("BC", axis = 1)
Y_train = train_df[["BC"]]
st = time.time()
svr_estimator, best_parameters, train_predicted_Y, MAE_train, MAE_validation = lib.train_SVR(X_train,Y_train)
et = time.time()
elapsed_time = (et - st) / 60
print('Execution time:', elapsed_time, ' minutes')
"""if best_parameters == 0:
    continue"""
R2_train = r2_score(Y_train, train_predicted_Y)
print("Best hyper-parameters found during the training: ", best_parameters)
print("MAE_train: ", MAE_train)
print("MAE_validation: ", MAE_validation)

#Testing
X_test = test_df.drop("BC", axis = 1)
Y_test = test_df[["BC"]]
test_predicted_Y = lib.test_SVR(X_test, Y_test, best_parameters)
MAE_test = mean_absolute_error(Y_test, test_predicted_Y)
R2_test = r2_score(Y_test, test_predicted_Y)
print("MAE_test: ", MAE_test)

print("R2_train: ", R2_train)
print("R2_test: ", R2_test)

#unscaled_train_Y, unscaled_train_predicted_Y = destandardize(Y_train, train_predicted_Y, scaler, nb_col)

unscaled_test_Y, unscaled_test_predicted_Y = destandardize(Y_test, test_predicted_Y, scaler, nb_col)
unscaled_MAE_test = mean_absolute_error(unscaled_test_Y, unscaled_test_predicted_Y)
print("unscaled MAE_test: ", unscaled_MAE_test)