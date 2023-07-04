import pandas as pd
from common_fct import remove_nan, print_nan_per, destandardize, split, filter_df, sample_split, BC_plot
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

##RANDOM FOREST
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
winter_df, pre_monsoon_df, summer_df, post_monsoon_df = split(df)

dates = {"winter": {"start":"2018-12-16", "end": "2018-12-20"}, "pre_monsoon":{"start": "2019-03-18", "end": "2019-03-24"}, "summer":{"start":"2019-06-20", "end": "2019-06-24"}, "post_monsoon": {"start": "2019-11-01", "end": "2019-11-04"}}

winter_sample_df = sample_split(dates["winter"]["start"], dates["winter"]["end"], df)
pre_monsoon_sample_df = sample_split(dates["pre_monsoon"]["start"], dates["pre_monsoon"]["end"], df)
summer_sample_df = sample_split(dates["summer"]["start"], dates["summer"]["end"], df)
post_monsoon_sample_df = sample_split(dates["post_monsoon"]["start"], dates["post_monsoon"]["end"], df)

seasons_df_list = [{"season": "winter", "sample_seasons_df": winter_sample_df, "seasons_df": winter_df}, 
{"season": "pre_monsoon", "sample_seasons_df": pre_monsoon_sample_df, "seasons_df": pre_monsoon_df}, 
{"season": "summer", "sample_seasons_df": summer_sample_df, "seasons_df": summer_df}, 
{"season": "post_monsoon", "sample_seasons_df": post_monsoon_sample_df, "seasons_df": post_monsoon_df}]

s = seasons_df_list[3]
season = s["season"]
#sample_seasons_df = s["sample_seasons_df"]
season_df = s["seasons_df"]
season_df = season_df.drop(["date","Hrs."], axis = 1)
season_df = remove_nan(season_df)
season_df = season_df.sample(frac=1, random_state=42) #shuffle
nb_col = len(season_df.columns)
print("--------------------------------------------------------------------------------------------------------------------------------------")
print("Season under study: ", season)
train, test = train_test_split(season_df, test_size=0.25, random_state=42) #by default the data is shuffled
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
rf_estimator, best_parameters, train_predicted_Y, MAE_train, MAE_validation = lib.train_RF(X_train,Y_train)
et = time.time()
elapsed_time = (et - st) / 60
#print('Execution time:', elapsed_time, ' minutes')
R2_train = r2_score(Y_train, train_predicted_Y)

"""if best_parameters == 0:
    continue"""
#print("Best hyper-parameters found during the training: ", best_parameters)
print("MAE_train: ", MAE_train)
print("MAE_validation_cv: ", MAE_validation)
#Testing
X_test = test_df.drop("BC", axis = 1)
Y_test = test_df[["BC"]]
test_predicted_Y = lib.test_RF(X_test, Y_test, best_parameters)
MAE_test = mean_absolute_error(Y_test, test_predicted_Y)
unscaled_test_Y, unscaled_test_predicted_Y = destandardize(Y_test, test_predicted_Y, scaler, nb_col)
unscaled_MAE_test = mean_absolute_error(unscaled_test_Y, unscaled_test_predicted_Y)
print("MAE_test: ", MAE_test)
print("unscaled MAE_test: ", unscaled_MAE_test)
R2_test = r2_score(Y_test, test_predicted_Y)
print("R2_train: ", R2_train)
print("R2_test: ", R2_test)
