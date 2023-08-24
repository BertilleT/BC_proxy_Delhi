"""

Author: Bertille Temple
Last update: August 22, 2023
Research group: Statistical Analysis of Networks and Systems SANS
Department: Computers Architecture Department DAC
Institution: Polytechnic University of Catalonia UPC

"""

import numpy as np
import pandas as pd 
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import warnings
import sys
import os

class Tune_trainer():
    # Per is the percentage of difference between training and validation scores we are willing to accept
    per = 0.06
    no_param_found = "In cv, we could not find hyper parameters for which the difference between error validation and error training is inferior to alpha = "
    
    # ignore mlp warnings (from the pipeline)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

    # Train the Random Forest model
    def train_RF(self, X, Y, scoring, best_params, std_all_training):   
        alpha = Tune_trainer.per * (Y.quantile(0.9))
        alpha = alpha.item()
        kfold = 10

        if best_params != 'null':
            n_estimators = [best_params[0]]
            max_features = [best_params[1]]
            max_depth = [best_params[2]]
        else: 
            n_estimators = [10, 100, 500]
            max_features = [1, 5, 10, 15, 20, 30]
            max_depth = [3, 5]

        param_grid = {'n_estimators' : n_estimators, 'max_features': max_features, 'max_depth': max_depth}

        rf_estimator = RandomForestRegressor()

        if std_all_training == False:
            scaler = preprocessing.StandardScaler()
            param_grid = {'model__' + key: value for key, value in param_grid.items()}
            pipe = Pipeline([('scaler', scaler), ('model', rf_estimator)]) 
            search = GridSearchCV(pipe, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True, n_jobs=2)
        else: 
            search = GridSearchCV(rf_estimator, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True, n_jobs=2)
        
        search.fit(X, np.ravel(Y))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)
        #the difference between the training and the validation score should not exceed a certain value, called alpha. This line was introduced to reduce the overfitting. 
        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < alpha else 0, axis = 1) 
        cv_scores_df = cv_scores_df.loc[cv_scores_df['keep'] == 1]    
        
        if len(cv_scores_df.index) == 0:
            print(Tune_trainer.no_param_found + str(alpha))
            return 0, 0, 0, 0, 0
        else: 
            best = cv_scores_df.loc[cv_scores_df["mean_test_score"].idxmax()]
            error_train = best["mean_train_score"]
            error_validation = best["mean_test_score"]
            if std_all_training == False:
                best_n = best['param_model__n_estimators']
                best_features = best['param_model__max_features']
                best_depth = best['param_model__max_depth']
            else:
                best_n = best['param_n_estimators']
                best_features = best['param_max_features']
                best_depth = best['param_max_depth']

            rf_estimator = RandomForestRegressor(n_estimators = best_n, max_features = best_features, max_depth = best_depth)
            if std_all_training == False:
                scaler = preprocessing.StandardScaler()
                X = scaler.fit_transform(X)
            rf_estimator.fit(X, np.ravel(Y))
            data_predict_train = rf_estimator.predict(X)

            return rf_estimator, [best_n, best_features, best_depth], data_predict_train, -error_train, -error_validation

    # Train the Support Vector Regression model
    def train_SVR(self, X, Y, scoring, best_params, std_all_training):
        alpha = Tune_trainer.per * (Y.quantile(0.9))
        alpha = alpha.item()
        kfold = 10

        if best_params != 'null':
            Cs = [best_params[0]]
            gammas = [best_params[1]]
            epsilons = [best_params[2]]
        else: 
            Cs = [1, 10, 100]
            gammas = [0.001, 0.01, 0.1]
            epsilons = [0.01, 0.1, 1]
            #kernels = ["rbf", "poly", "sigmoid"]

        param_grid = { 'C' : Cs, 'gamma': gammas, 'epsilon': epsilons}#, 'kernel': kernels}
        
        svr_estimator = svm.SVR()

        if std_all_training == False:
            scaler = preprocessing.StandardScaler()
            param_grid = {'model__' + key: value for key, value in param_grid.items()}
            pipe = Pipeline([('scaler', scaler), ('model', svr_estimator)]) 
            search = GridSearchCV(pipe, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True, n_jobs=2)
        else: 
            search = GridSearchCV(svr_estimator, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True, n_jobs=2)
        
        search.fit(X, np.ravel(Y))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)

        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < alpha else 0, axis = 1)
        cv_scores_df = cv_scores_df.loc[cv_scores_df['keep'] == 1]

        if len(cv_scores_df.index) == 0:
            print(Tune_trainer.no_param_found + str(alpha))
            return 0, 0, 0, 0, 0
        else: 
            best = cv_scores_df.loc[cv_scores_df["mean_test_score"].idxmax()]
            error_train = best["mean_train_score"]
            error_validation = best["mean_test_score"]
            if std_all_training == False:
                best_c = best['param_model__C']
                best_gamma = best['param_model__gamma']
                best_eps = best['param_model__epsilon']
            else:
                best_c = best['param_C']
                best_gamma = best['param_gamma']
                best_eps = best['param_epsilon']
            
            svr_estimator = svm.SVR(C = best_c, gamma = best_gamma, epsilon = best_eps)
            if std_all_training == False:
                scaler = preprocessing.StandardScaler()
                X = scaler.fit_transform(X)
            svr_estimator.fit(X, np.ravel(Y))
            data_predict_train = svr_estimator.predict(X)
            
            return svr_estimator, [best_c, best_gamma, best_eps], data_predict_train, -error_train, -error_validation
    
    # Train the Neural Networks model, which is a Multi Layer Perceptron
    def train_NN(self, X, Y, scoring, best_params, std_all_training): 
        alpha = Tune_trainer.per * (Y.quantile(0.9))
        alpha = alpha.item()
        kfold = 10

        if best_params != 'null':
            nb_neurons = [best_params[0]]
            activation = [best_params[1]]
            optimizer = [best_params[2]]
            alpha_nn = [best_params[3]]
            learning_rate = [best_params[4]]
        else: 
            nb_neurons = [(50,), (100, 50), (100, 100, 50)]
            activation = ['relu']
            optimizer = ['adam']
            alpha_nn = [0.0001, 0.001, 0.01]
            learning_rate = ['constant']
        
        param_grid = { 'hidden_layer_sizes' : nb_neurons, 'activation': activation, 'solver': optimizer, 'alpha': alpha_nn, 'learning_rate': learning_rate}
        
        mlp_estimator = MLPRegressor(random_state=1, max_iter=50)

        if std_all_training == False:
            scaler = preprocessing.StandardScaler()
            param_grid = {'model__' + key: value for key, value in param_grid.items()}
            pipe = Pipeline([('scaler', scaler), ('model', mlp_estimator)]) 
            search = GridSearchCV(pipe, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True, n_jobs=2)
        else: 
            search = GridSearchCV(mlp_estimator, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True, n_jobs=2)
        
        search.fit(X, np.ravel(Y))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)

        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < alpha else 0, axis = 1)     
        cv_scores_df = cv_scores_df.loc[cv_scores_df['keep'] == 1]   

        if len(cv_scores_df.index) == 0:
            print(Tune_trainer.no_param_found + str(alpha))
            return 0, 0, 0, 0, 0
        else: 
            best = cv_scores_df.loc[cv_scores_df["mean_test_score"].idxmax()]
            error_train = best["mean_train_score"]
            error_validation = best["mean_test_score"]
            if std_all_training == False:
                best_hidden_layer_sizes = best['param_model__hidden_layer_sizes']
                best_activation = best['param_model__activation']
                best_solver = best['param_model__solver']
                best_alpha = best['param_model__alpha']
                best_learning_rate = best['param_model__learning_rate']
            else:
                best_hidden_layer_sizes = best['param_hidden_layer_sizes']
                best_activation = best['param_activation']
                best_solver = best['param_solver']
                best_alpha = best['param_alpha']
                best_learning_rate = best['param_learning_rate']

            mlp_estimator = MLPRegressor(hidden_layer_sizes = best_hidden_layer_sizes, activation = best_activation, solver = best_solver, alpha = best_alpha, learning_rate = best_learning_rate)
            if std_all_training == False:
                scaler = preprocessing.StandardScaler()
                X = scaler.fit_transform(X)
            mlp_estimator.fit(X, np.ravel(Y))
            data_predict_train = mlp_estimator.predict(X)

            return mlp_estimator, [best_hidden_layer_sizes, best_activation, best_solver, best_alpha, best_learning_rate], data_predict_train, -error_train, -error_validation