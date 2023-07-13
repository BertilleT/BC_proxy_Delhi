import pandas as pd
from predict_BC import train_test_ML
from predict_BC_class import predict_BC_lib

## --------------- DEFINING VARIABLES --------------- ##

data_path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
RH_path = '../data/Relative humidity data_Delhi 2018-2019.csv'
method = 'SVR' 
RH_included = True
RH_imputed = False
std_all_training = False #True if the whole training is standardized before cross validation. Else, false. When it is true, there is data leakage. 
tune_hyperparameters = False #True if we want to tune again the hyper parameters. False if we eant to use the ones already found. 

#dictonnaries with the best hyper parameters already found during the training. 
best_param = {}
if method == 'SVR':
    if RH_included ==  True: 
        best_param["whole_dataset"] = [10, 0.1, 0.1]
        best_param["winter"] = [100, 0.1, 0.01] 
        best_param["pre_monsoon"] = [10, 0.1, 0.1] 
        best_param["summer"] = [100, 0.1, 0.01]
        best_param["post_monsoon"] = [10, 0.1, 0.01] 
    elif RH_included ==  False:
        best_param["whole_dataset"] = [10, 0.1, 0.1] 
        best_param["winter"] = [100, 0.1, 0.1]
        best_param["pre_monsoon"] = [10, 0.1, 0.1]
        best_param["summer"] = [10, 0.1, 0.1]
        best_param["post_monsoon"] = [10, 0.1, 0.1]
elif method == 'RF':
    best_param["whole_dataset"] = [500, 10, 5]
    best_param["winter"] = [10, 1, 3]
    best_param["pre_monsoon"] = [500, 30, 3]
    best_param["summer"] = [500, 15, 3]
    best_param["post_monsoon"] = [500, 20, 3]
elif method == 'NN':
    best_param["whole_dataset"] = [(100, 100, 50), 'relu', 'adam', 0.0001, 'constant']#[(100, 100, 50), 'relu', 'adam', 0.01, 'constant']
    best_param["winter"] = [(100, 50), 'relu', 'adam', 0.001, 'constant']
    best_param["pre_monsoon"] = [(100, 50), 'relu', 'adam', 0.0001, 'constant']
    best_param["summer"] = [(50,), 'relu', 'adam', 0.001, 'constant']
    best_param["post_monsoon"] = [(100, 100, 50), 'relu', 'adam', 0.0001, 'constant']

if tune_hyperparameters == True:
    for key in my_dict:
        my_dict[key] = 'null'
scoring = 'neg_root_mean_squared_error'
#scoring = 'neg_mean_absolute_error' 


## --------------- PROCESSING --------------- ##

lib = predict_BC_lib()
print("Method used: ", method)
print("RH feature included: ", RH_included)
print("RH null values imputed: ", RH_imputed)
print("Is the training scaled before cross validation (bad approach) ? ", std_all_training)
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

test_true_values, test_predictions, metrics = train_test_ML(df, method, scoring, "whole_dataset", best_param, std_all_training)
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
    print("-----------------------------------------------------------------------------------------------------------------")
    print("Season under study: ", season)
    df = season_df
    df = lib.remove_nan_columns(df, RH_included)
    df = lib.remove_nan_rows(df)
    df, datetime_df = lib.concat_date_time(df)
    test_true_values, test_predictions, metrics = train_test_ML(df, method, scoring, season, best_param, std_all_training)
    print(metrics)
    #lib.trueANDpred_time_plot(test_true_values, test_predictions, datetime_df, method, season)
    #lib.trueVSpred_scatter_plot(test_true_values, test_predictions, method, season)