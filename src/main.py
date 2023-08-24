"""

Author: Bertille Temple
Last update: August 22, 2023
Research group: Statistical Analysis of Networks and Systems SANS
Department: Computers Architecture Department DAC
Institution: Polytechnic University of Catalonia UPC

"""

import pandas as pd
# Import classes
from pre_post_process import Pre_post_process
from plot import Plot
# Import function
from predict_BC import predict_BC
#Import parameters
from parameters import *

pre_post_process = Pre_post_process()
plot = Plot()

print("Method used: ", method)
print("RH feature included: ", RH_included)
print("RH null values imputed: ", RH_imputed)
print("Training scaled before cross validation: ", std_all_training)

## ------------------------------ Whole_dataset ------------------------------ ##

df = pd.read_excel(data_path)
#print the percentage of nan values per column
pre_post_process.print_nan_per(df)

if RH_imputed == True:
    rh = pd.read_csv(RH_path)
    #plot.RH(df, rh)
    df = pre_post_process.impute_RH(df, rh)

if SR_included == True:
    sr = pd.read_csv(SR_path)
    df = pre_post_process.concat_SR(df, sr)

#remove columns with 99 or 100% of nan values. If RH_included was set to False, remove RH column too. 
df = pre_post_process.remove_nan_columns(df, RH_included)
#in the resulting df, remove the rows with missing values
df = pre_post_process.remove_nan_rows(df)
df, datetime_df = pre_post_process.concat_date_time(df)
plot.season_split(df, RH_included, RH_imputed)

test_true_values, test_predictions, metrics = predict_BC(df, method, scoring, "whole_dataset", best_param, std_all_training)
print(metrics)
plot.trueANDpred_time(test_true_values, test_predictions, datetime_df, method, "whole_dataset", save_images)
plot.trueVSpred_scatter(test_true_values, test_predictions, method, "whole_dataset", save_images)

## ------------------------------ Seasonal subset ------------------------------ ##

df = pd.read_excel(data_path)
if RH_imputed == True:
    rh = pd.read_csv(RH_path)
    df = pre_post_process.impute_RH(df, rh)

if SR_included == True:
    sr = pd.read_csv(SR_path)
    df = pre_post_process.concat_SR(df, sr)

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
#split df into seasons
winter_df, pre_monsoon_df, summer_df, post_monsoon_df = pre_post_process.split(df)
seasons_dict = {"winter": winter_df, "pre_monsoon": pre_monsoon_df, "summer": summer_df, "post_monsoon": post_monsoon_df}

for season, season_df in seasons_dict.items(): 
    print("-----------------------------------------------------------------------------------------------------------------")
    print("Season under study: ", season)
    df = season_df
    df = pre_post_process.remove_nan_columns(df, RH_included)
    df = pre_post_process.remove_nan_rows(df)
    df, datetime_df = pre_post_process.concat_date_time(df)
    test_true_values, test_predictions, metrics = predict_BC(df, method, scoring, season, best_param, std_all_training)
    print(metrics)
    plot.trueANDpred_time(test_true_values, test_predictions, datetime_df, method, season, save_images)
    plot.trueVSpred_scatter(test_true_values, test_predictions, method, season, save_images)