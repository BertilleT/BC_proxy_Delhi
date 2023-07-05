import pandas as pd
from predict_BC import train_test_ML
from predict_BC_class import predict_BC_lib

## --------------- DEFINING VARIABLES --------------- ##

data_path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
method = 'SVR' 
#method = 'RF'
#dictonnaries with the best hyper parameters already found during the training. 
best_param = {}
if method == 'SVR': 
    best_param["whole dataset"] = [10, 0.1, 0.1]#[10, 0.1, 0.1] WITH RH
    best_param["winter"] = [100, 0.1, 0.1]#[100, 0.1, 0.01] WITH RH
    best_param["pre_monsoon"] = [10, 0.1, 0.1]#[10, 0.1, 0.1] WITH RH
    best_param["summer"] = [10, 0.1, 0.1]#[100, 0.1, 0.01] WITH RH
    best_param["post_monsoon"] = [10, 0.1, 0.1]#[10, 0.1, 0.01] WITH RH
elif method == 'RF':
    best_param["whole dataset"] = [500, 10, 5]
    best_param["winter"] = [10, 1, 3]
    best_param["pre_monsoon"] = [500, 30, 3]
    best_param["summer"] = [500, 15, 3]
    best_param["post_monsoon"] = [500, 20, 3]
elif method == 'NN':
    best_param["whole dataset"] = [2, 0.001, 5, 32]

#scoring = 'neg_mean_squared_error'
scoring = 'neg_mean_absolute_error' 

## --------------- PROCESSING --------------- ##
lib = predict_BC_lib()
## Whole dataset 
df = pd.read_excel(data_path)
df = lib.remove_nan(df)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
metrics = train_test_ML(df, method, scoring, "whole dataset", best_param)
print(metrics)

## Seasonal subset 
#split df into seasons
df = pd.read_excel(data_path)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
winter_df, pre_monsoon_df, summer_df, post_monsoon_df = lib.split(df)
seasons_dict = {"winter": winter_df, "pre_monsoon": pre_monsoon_df, "summer": summer_df, "post_monsoon": post_monsoon_df}
#seasons_dict = {"winter": winter_df, "summer": summer_df}
for season, season_df in seasons_dict.items(): 
    print("--------------------------------------------------------------------------------------------------------------------------------------")
    print("Season under study: ", season)
    df = season_df
    df = lib.remove_nan(df)
    metrics = train_test_ML(df, method, scoring, season, best_param)
    print(metrics)