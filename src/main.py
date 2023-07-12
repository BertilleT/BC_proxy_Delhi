import pandas as pd
from predict_BC import train_test_ML
from predict_BC_class import predict_BC_lib

## --------------- DEFINING VARIABLES --------------- ##

data_path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
RH_path = '../data/Relative humidity data_Delhi 2018-2019.csv'
method = 'RF' 
RH_included = True
RH_imputed = True

#dictonnaries with the best hyper parameters already found during the training. 
best_param = {}
if method == 'SVR': 
    best_param["whole_dataset"] = [10, 0.1, 0.1]#[10, 0.1, 0.1] #WITH RH #[10, 0.1, 0.1] WITHOUT RH
    best_param["winter"] = [100, 0.1, 0.01] #[100, 0.1, 0.1]#[100, 0.1, 0.01] WITH RH
    best_param["pre_monsoon"] = [10, 0.1, 0.1] #[10, 0.1, 0.1]#[10, 0.1, 0.1] WITH RH
    best_param["summer"] = [100, 0.1, 0.01]#[10, 0.1, 0.1]#[100, 0.1, 0.01] WITH RH
    best_param["post_monsoon"] = [10, 0.1, 0.01] #[10, 0.1, 0.1]#[10, 0.1, 0.01] WITH RH
elif method == 'RF':
    best_param["whole_dataset"] = [500, 10, 5]
    best_param["winter"] = [10, 1, 3]
    best_param["pre_monsoon"] = [500, 30, 3]
    best_param["summer"] = [500, 15, 3]
    best_param["post_monsoon"] = [500, 20, 3]
elif method == 'NN':
    #IVA best_param["whole dataset"] = [2, 0.001, 5, 32]
    best_param["whole_dataset"] = [(100, 100, 50), 'relu', 'adam', 0.0001, 'constant']#[(100, 100, 50), 'relu', 'adam', 0.01, 'constant']
    best_param["winter"] = [(100, 50), 'relu', 'adam', 0.001, 'constant']
    best_param["pre_monsoon"] = [(100, 50), 'relu', 'adam', 0.0001, 'constant']
    best_param["summer"] = [(50,), 'relu', 'adam', 0.001, 'constant']
    best_param["post_monsoon"] = [(100, 100, 50), 'relu', 'adam', 0.0001, 'constant']

#scoring = 'neg_root_mean_squared_error'
scoring = 'neg_mean_absolute_error' 


## --------------- PROCESSING --------------- ##

lib = predict_BC_lib()

## Whole_dataset 

df = pd.read_excel(data_path)
if RH_imputed == True:
    rh = pd.read_csv(RH_path)
    #lib.plot_RH(df, rh)
    df = lib.impute_RH(df, rh)

df = lib.remove_nan_columns(df, RH_included)
df = lib.remove_nan_rows(df)
df, datetime_df = lib.concat_date_time(df)
#lib.season_split_plot(df)

test_true_values, test_predictions, metrics = train_test_ML(df, method, scoring, "whole_dataset", best_param)
print(metrics)
#lib.trueANDpred_time_plot(test_true_values, test_predictions, datetime_df, method, "whole_dataset")
#lib.trueVSpred_scatter_plot(test_true_values, test_predictions, method, "whole_dataset")

## Seasonal subset 

#split df into seasons
df = pd.read_excel(data_path)
if RH_imputed == True:
    rh = pd.read_csv(RH_path)
    df = lib.impute_RH(df, rh)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
winter_df, pre_monsoon_df, summer_df, post_monsoon_df = lib.split(df)
seasons_dict = {"winter": winter_df, "pre_monsoon": pre_monsoon_df, "summer": summer_df, "post_monsoon": post_monsoon_df}

for season, season_df in seasons_dict.items(): 
    print("--------------------------------------------------------------------------------------------------------------------------------------")
    print("Season under study: ", season)
    df = season_df
    df = lib.remove_nan_columns(df, RH_included)
    df = lib.remove_nan_rows(df)
    df, datetime_df = lib.concat_date_time(df)
    test_true_values, test_predictions, metrics = train_test_ML(df, method, scoring, season, best_param)
    print(metrics)
    #lib.trueANDpred_time_plot(test_true_values, test_predictions, datetime_df, method, season)
    #lib.trueVSpred_scatter_plot(test_true_values, test_predictions, method, season)