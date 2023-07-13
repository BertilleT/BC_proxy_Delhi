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

    #the index is usefull to recover temporality and draw plots, so let us store it. 
    df['index'] = df.index
    X = df.drop("BC", axis = 1)
    y = df[["index", "BC"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train.set_index('index', inplace=True)
    y_train.set_index('index', inplace=True)
    X_test.set_index('index', inplace=True)
    y_test.set_index('index', inplace=True)

    #standardize
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    #scaling the training (not yet separated from the validation) would results in data leakage. Remind that we use cross validation.  
    if std_all_training == True:
        print("ERROR, SHOULD NOT BE DONE")
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    #---------------------------------------------------------------------------------------------------------------------#
    ## Training
    st = time.time()
    if method == 'SVR':
        model, best_parameters, train_predicted_Y, error_train, error_validation = lib.train_SVR(X_train, y_train, scoring, best_parameters[season], std_all_training)
    elif method == 'RF':
        model, best_parameters, train_predicted_Y, error_train, error_validation = lib.train_RF(X_train, y_train, scoring, best_parameters[season], std_all_training)
    elif method == 'NN':
        model, best_parameters, train_predicted_Y, error_train, error_validation = lib.train_NN(X_train, y_train, scoring, best_parameters[season], std_all_training)
    et = time.time()
    elapsed_time = (et - st) / 60
    if (model, best_parameters, train_predicted_Y, error_train, error_validation) == (0, 0, 0, 0, 0):
        return 'Fail in the training. Change the hyper parameters, or increase the value of alpha.'
    
    #R2 train unscaled data
    R2_train = r2_score(y_train, train_predicted_Y)
    #print('Execution time:', elapsed_time, ' minutes')
    #feature_importances_train = pd.DataFrame(model.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance',ascending=False)
    
    #---------------------------------------------------------------------------------------------------------------------#
    ## Testing

    if method == 'SVR':
        model = SVR(C = best_parameters[0], gamma = best_parameters[1], epsilon = best_parameters[2])
        model.fit(X_test, np.ravel(y_test))
        predicted_y_test = model.predict(X_test)    
    elif method == 'RF':
        model = RandomForestRegressor(n_estimators = best_parameters[0], max_features = best_parameters[1], max_depth = best_parameters[2])
        model.fit(X_test, np.ravel(y_test))
        predicted_y_test = model.predict(X_test)   
    elif method == 'NN':
        model = MLPRegressor(hidden_layer_sizes = best_parameters[0], activation = best_parameters[1], solver = best_parameters[2], alpha = best_parameters[3], learning_rate = best_parameters[4])
        model.fit(X_test, np.ravel(y_test))
        predicted_y_test = model.predict(X_test)     

    if scoring == 'neg_root_mean_squared_error':
        #error_test = np.sqrt(mean_squared_error(y_test, predicted_y_test))
        error_test = np.sqrt(mean_squared_error(y_test, predicted_y_test))
    elif scoring == 'neg_mean_squared_error':
        #error_test = mean_squared_error(y_test, predicted_y_test)
        error_test = mean_squared_error(y_test, predicted_y_test)
    elif scoring == 'neg_mean_absolute_error':
        #error_test = mean_absolute_error(y_test, predicted_y_test)
        error_test = mean_absolute_error(y_test, predicted_y_test)
    
    
    predicted_y_test = pd.DataFrame(predicted_y_test, index = X_test.index, columns=["BC"])

    R2_test = r2_score(y_test, predicted_y_test)

    metrics['best_parameters'] = best_parameters
    metrics['error_train'] = round(error_train,2)
    metrics['error_validation'] = round(error_validation, 2)
    #metrics['error_test'] = round(error_test, 2)
    metrics['error_test'] = round(error_test, 2)
    metrics['R2_train'] = round(R2_train, 2)
    metrics['R2_test'] = round(R2_test, 2)

    return y_test, predicted_y_test, metrics