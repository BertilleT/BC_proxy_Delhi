import pandas as pd
from predict_BC_class import predict_BC_lib
import numpy as np
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
pd.options.mode.chained_assignment = None

lib = predict_BC_lib()

def train_test_ML(df, method, scoring, season, best_parameters, std_all_training):
    #---------------------------------------------------------------------------------------------------------------------#
    ## Pre-processing
    metrics = {}
    df = df.drop(["date","Hrs.", "datetime"], axis = 1)
    #store the number of features
    nb_col = len(df.columns)
    print("95% of the BC values of the set under study are between " + str(df["BC"].min().round()) + " and " + str(df["BC"].quantile(0.95).round()))
    print("The mean is: " +  str(df["BC"].mean().round()))

    X = df.drop("BC", axis = 1)
    y = df[["BC"]]
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #the index is usefull to recover temporality and draw plots, so let us store it. 
    train_index = train_df.index
    test_index = test_df.index

    #standardize
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_df)
    #scaling the training (not yet separated from the validation) would results in data leakage. Remind that we use cross validation.  
    if std_all_training == True:
        train_df = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns, index=train_index)

    test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_index)

    #---------------------------------------------------------------------------------------------------------------------#
    ## Training

    X_train = train_df.drop("BC", axis = 1)
    Y_train = train_df[["BC"]]
    st = time.time()
    if method == 'SVR':
        model, best_parameters, train_predicted_Y, error_train, error_validation = lib.train_SVR(X_train, Y_train, scoring, best_parameters[season], std_all_training)
    elif method == 'RF':
        model, best_parameters, train_predicted_Y, error_train, error_validation = lib.train_RF(X_train, Y_train, scoring, best_parameters[season])
    elif method == 'NN':
        model, best_parameters, train_predicted_Y, error_train, error_validation = lib.train_NN(X_train, Y_train, scoring, best_parameters[season])
    et = time.time()
    elapsed_time = (et - st) / 60
    if (model, best_parameters, train_predicted_Y, error_train, error_validation) == (0, 0, 0, 0, 0):
        return 'Fail in the training. Change the hyper parameters, or increase the value of alpha.'
    
    if std_all_training == True:
        Y_train, train_predicted_Y = lib.destandardize(Y_train, train_predicted_Y, scaler, nb_col)

    #R2 train unscaled data
    R2_train = r2_score(Y_train, train_predicted_Y)
    #print('Execution time:', elapsed_time, ' minutes')
    #feature_importances_train = pd.DataFrame(model.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance',ascending=False)
    
    #---------------------------------------------------------------------------------------------------------------------#
    ## Testing
    X_test = test_df.drop("BC", axis = 1)
    Y_test = test_df[["BC"]]

    if method == 'SVR':
        model = SVR(C = best_parameters[0], gamma = best_parameters[1], epsilon = best_parameters[2])
        model.fit(X_test, np.ravel(Y_test))
        test_predicted_Y = model.predict(X_test)    
    elif method == 'RF':
        model = RandomForestRegressor(n_estimators = best_parameters[0], max_features = best_parameters[1], max_depth = best_parameters[2])
        model.fit(X_test, np.ravel(Y_test))
        test_predicted_Y = model.predict(X_test)   
    elif method == 'NN':
        model = MLPRegressor(hidden_layer_sizes = best_parameters[0], activation = best_parameters[1], solver = best_parameters[2], alpha = best_parameters[3], learning_rate = best_parameters[4])
        model.fit(X_test, np.ravel(Y_test))
        test_predicted_Y = model.predict(X_test)     

    unscaled_test_Y, unscaled_test_predicted_Y = lib.destandardize(Y_test, test_predicted_Y, scaler, nb_col)
    
    if scoring == 'neg_root_mean_squared_error':
        #error_test = np.sqrt(mean_squared_error(Y_test, test_predicted_Y))
        unscaled_error_test = np.sqrt(mean_squared_error(unscaled_test_Y, unscaled_test_predicted_Y))
    elif scoring == 'neg_mean_squared_error':
        #error_test = mean_squared_error(Y_test, test_predicted_Y)
        unscaled_error_test = mean_squared_error(unscaled_test_Y, unscaled_test_predicted_Y)
    elif scoring == 'neg_mean_absolute_error':
        #error_test = mean_absolute_error(Y_test, test_predicted_Y)
        unscaled_error_test = mean_absolute_error(unscaled_test_Y, unscaled_test_predicted_Y)
    
    R2_test = r2_score(Y_test, test_predicted_Y)

    metrics['best_parameters'] = best_parameters
    metrics['error_train'] = round(error_train,2)
    metrics['error_validation'] = round(error_validation, 2)
    #metrics['error_test'] = round(error_test, 2)
    metrics['error_test'] = round(unscaled_error_test, 2)
    metrics['R2_train'] = round(R2_train, 2)
    metrics['R2_test'] = round(R2_test, 2)

    return unscaled_test_Y, unscaled_test_predicted_Y, metrics