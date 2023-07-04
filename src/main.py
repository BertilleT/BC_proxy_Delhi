from predict_BC import train_test_ML

data_path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
method = 'SVR' 
#method = 'RF'
#scoring = 'neg_mean_absolute_error' 
scoring = 'neg_mean_squared_error'
season = "whole dataset"

#dictonnaries with the best hyper parameters already found during the training. 
SVR_best_param = {}
SVR_best_param["whole"] = [10, 0.1, 0.1]
SVR_best_param["winter"] = [100, 0.1, 0.01]
SVR_best_param["pre_monsoon"] = [10, 0.1, 0.1]
SVR_best_param["summer"] = [100, 0.1, 0.01]
SVR_best_param["post_monsoon"] = [10, 0.1, 0.01]

RF_best_param = {}
RF_best_param["whole"] = [500, 10, 5]
RF_best_param["winter"] = [10, 1, 3]
RF_best_param["pre_monsoon"] = [500, 30, 3]
RF_best_param["summer"] = [500, 15, 3]
RF_best_param["post_monsoon"] = [500, 20, 3]

metrics = train_test_ML(data_path, method, scoring, season, SVR_best_param)
print(metrics)
