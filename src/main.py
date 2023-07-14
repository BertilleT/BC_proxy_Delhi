import pandas as pd
from predict_BC import train_test_ML
from predict_BC_class import predict_BC_lib

## --------------- DEFINING VARIABLES --------------- ##

data_path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
RH_path = '../data/Relative humidity data_Delhi 2018-2019.csv'
SR_path = '../data/Solar radiation data_Agra 2017-2020.csv'

method = 'NN' 
RH_included = True
SR_included = True
RH_imputed = True
tune_hyperparameters = True #True if we want to tune again the hyper parameters. False if we eant to use the ones already found. 
save_images = True
scoring = 'neg_root_mean_squared_error'
#scoring = 'neg_mean_absolute_error' 

std_all_training = False #True if the whole training is standardized before cross validation. Should be False. 

#dictonnaries with the best hyper parameters already found during the training. 
best_param = {}
if method == 'SVR':
    best_param["whole_dataset"] = [100, 0.1, 1] #[10, 0.1, 0.1]
    best_param["winter"] = [100, 0.1, 1] #[100, 0.1, 0.1]
    best_param["pre_monsoon"] = [100, 0.1, 1] #[10, 0.1, 0.1]
    best_param["summer"] = [100, 0.1, 1] #[10, 0.1, 0.1]
    best_param["post_monsoon"] = [100, 0.1, 1] #[10, 0.1, 0.1]
elif method == 'RF':
    best_param["whole_dataset"] = [500, 20, 5]#[500, 10, 5] #OLD
    best_param["winter"] = [100, 20, 5]#[10, 1, 3] #OLD
    best_param["pre_monsoon"] = [100, 5, 5]#[500, 30, 3] #OLD
    best_param["summer"] = [100, 20, 5]#[500, 15, 3] #OLD
    best_param["post_monsoon"] = [100, 20, 5]#[500, 20, 3] #OLD
elif method == 'NN':
    best_param["whole_dataset"] = [(100, 100, 50), 'relu', 'adam', 0.001, 'constant'] #[(100, 100, 50), 'relu', 'adam', 0.0001, 'constant']
    best_param["winter"] = [(100, 100, 50), 'relu', 'adam', 0.001, 'constant']#[(100, 50), 'relu', 'adam', 0.001, 'constant']
    best_param["pre_monsoon"] = [(100, 100, 50), 'relu', 'adam', 0.0001, 'constant']#[(100, 50), 'relu', 'adam', 0.0001, 'constant']
    best_param["summer"] = [(100, 100, 50), 'relu', 'adam', 0.01, 'constant']#[(50,), 'relu', 'adam', 0.001, 'constant']
    best_param["post_monsoon"] = [(100, 100, 50), 'relu', 'adam', 0.001, 'constant']#[(100, 100, 50), 'relu', 'adam', 0.0001, 'constant']

if tune_hyperparameters == True:
    for key in best_param:
        best_param[key] = 'null'

## --------------- PROCESSING --------------- ##

lib = predict_BC_lib()
print("Method used: ", method)
print("RH feature included: ", RH_included)
print("RH null values imputed: ", RH_imputed)
print("Training scaled before cross validation: ", std_all_training)
## Whole_dataset 

df = pd.read_excel(data_path)
if RH_imputed == True:
    rh = pd.read_csv(RH_path)
    #lib.plot_RH(df, rh)
    df = lib.impute_RH(df, rh)

if SR_included == True:
    sr = pd.read_csv(SR_path)
    df = lib.concat_SR(df, sr)

df = lib.remove_nan_columns(df, RH_included)
df = lib.remove_nan_rows(df)
df, datetime_df = lib.concat_date_time(df)
lib.season_split_plot(df, RH_included, RH_imputed)

test_true_values, test_predictions, metrics = train_test_ML(df, method, scoring, "whole_dataset", best_param, std_all_training)
print(metrics)
lib.trueANDpred_time_plot(test_true_values, test_predictions, datetime_df, method, "whole_dataset", save_images)
lib.trueVSpred_scatter_plot(test_true_values, test_predictions, method, "whole_dataset", save_images)

## Seasonal subset 

#split df into seasons
df = pd.read_excel(data_path)
if RH_imputed == True:
    rh = pd.read_csv(RH_path)
    df = lib.impute_RH(df, rh)

if SR_included == True:
    sr = pd.read_csv(SR_path)
    df = lib.concat_SR(df, sr)
    
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
    lib.trueANDpred_time_plot(test_true_values, test_predictions, datetime_df, method, season, save_images)
    lib.trueVSpred_scatter_plot(test_true_values, test_predictions, method, season, save_images)