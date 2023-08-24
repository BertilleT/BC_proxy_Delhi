## --------------- DEFINING PARAMETERS --------------- ##

data_path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
RH_path = '../data/Relative humidity data_Delhi 2018-2019.csv'
SR_path = '../data/Solar radiation data_Agra 2017-2020.csv' #Note that for now, we use the Solar Radiation measures from Agra. 

#Which method should be used to predict Black Carbon ?
method = 'NN' #'SVR', or 'RF'

#Should the original Relative Humidity measures be used ?
RH_included = True #False. When it is set to False, we avoid dropping 25% of the dataset

#Should the Solar Radiation be a feature ? 
SR_included = True #False

#Should the Relative Humidity be imputed with the new measures obtained from "Relative humidity data_Delhi 2018-2019" file, when they are missing in the original file ? 
RH_imputed = False #True

#Should the hyper-parameters be tuned ? 
tune_hyperparameters = False #True if we want to tune again the hyper parameters. False if we want to use the ones already found previously. 

#Should images of the results be saved ?
save_images = False

#Which score should be used to tune the hyper-parameters and or to asses the quality of the predictions ? 
scoring = 'neg_root_mean_squared_error' # 'neg_mean_absolute_error' 

#Should the training set be standardize ? 
std_all_training = False #True if the whole training is standardized before cross validation(incorrect). It should be False.

## --------------- HYPER-PARAMETERS ALREADY TUNED --------------- ##

## Dictionaries with the best hyper parameters already found during the training. 
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