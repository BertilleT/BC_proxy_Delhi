"authors: Pau Ferrer Cid and Bertille Temple"

import numpy as np
import pandas as pd 
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score

class BC_proxy_lib():
    def train_tune_RF(self, X_des, Y_des):
        kfold = 10
        n_estimators = [50, 500]
        max_features = [3, 5, 10, 15]
        max_depth = [1, 5, 15, 20]
        param_grid = { 'n_estimators' : n_estimators, 'max_features': max_features, 'max_depth': max_depth}
        rf_estimator = RandomForestRegressor()
        search = GridSearchCV(rf_estimator, scoring='neg_mean_squared_error', param_grid = param_grid, cv = kfold, refit = False, return_train_score=True)
        search.fit(X_des, np.ravel(Y_des))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)
        #print(cv_scores_df[["mean_train_score", "mean_test_score"]].to_string())
        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < 0.06 else 0, axis = 1)
        #cv_scores_df[["param_n_estimators",	"param_max_features",	"param_max_depth",	"params", "mean_train_score", "mean_test_score", "keep"]].to_csv('../scores/0.06_all_cv_scores_rf_without_PM10.csv')
        #print(cv_scores_df[["param_n_estimators", "param_max_features", "param_max_depth", "keep"]].to_string())
        #n_estimators_filter_df is made to filter the possible values of n_estimators studied in the gridsearch. 
        #we filter the values of n_estimators for which the keep is always false, 
        #which mean for which the difference between mean rmse train and mean rmse validation is always >= 0.6, whatever the other hyper parameters. 
        n_estimators_filter_df = cv_scores_df.groupby(["param_n_estimators"], as_index = False)["keep"].sum()
        n_estimators_filter_df["n_to_filter"] = n_estimators_filter_df.apply(lambda x : 1 if x.keep == 0 else 0, axis=1)
        #print(n_estimators_filter_df)
        max_features_filter_df = cv_scores_df.groupby(["param_max_features"], as_index = False)["keep"].sum()
        max_features_filter_df["features_to_filter"] = max_features_filter_df.apply(lambda x : 1 if x.keep == 0 else 0, axis=1)
        #print(max_features_filter_df)
        max_depth_filter_df = cv_scores_df.groupby(["param_max_depth"], as_index = False)["keep"].sum()
        max_depth_filter_df["depth_to_filter"] = max_depth_filter_df.apply(lambda x : 1 if x.keep == 0 else 0, axis=1)
        #print(max_depth_filter_df)
        cv_scores_df = cv_scores_df.loc[cv_scores_df['keep'] == 1]
        if len(cv_scores_df.index) == 0:
            print("In cv, we could not find hyper parameters for which rmse validation and rmse training are close enough" )
            return 0, 0, 0, 0, 0
        else: 
            best = cv_scores_df.loc[cv_scores_df["mean_test_score"].idxmax()]
            RMSE_train = best["mean_train_score"]
            RMSE_validation = best["mean_test_score"]

            best_n = best['param_n_estimators']#500
            best_features = best['param_max_features']#10
            best_depth = best['param_max_depth']#5
            rf_estimator = RandomForestRegressor(n_estimators = best_n, max_features = best_features, max_depth = best_depth)
            rf_estimator.fit(X_des, np.ravel(Y_des))
            data_predict_train = rf_estimator.predict(X_des)
            return rf_estimator, [best_n, best_features, best_depth], data_predict_train, -RMSE_train, -RMSE_validation

    def test_RF(self, X_des, Y_des, best_parameters):
        model = RandomForestRegressor(n_estimators = best_parameters[0], max_features = best_parameters[1], max_depth = best_parameters[2])
        model.fit(X_des, np.ravel(Y_des))
        data_predict_test = model.predict(X_des)    
        return data_predict_test

    def train_tune_SVR(self, X_des, Y_des):
        kfold = 10
        cs = [10]
        gs = [0.1]
        epsilons = [0.1]
        #kernels = ["rbf", "poly", "sigmoid"]
        param_grid = { 'C' : cs, 'gamma': gs, 'epsilon': epsilons}#, 'kernel': kernels}
        svr_estimator = svm.SVR()
        search = GridSearchCV(svr_estimator, scoring = 'neg_mean_squared_error',param_grid = param_grid, cv = kfold, refit = False, return_train_score=True)
        search.fit(X_des, np.ravel(Y_des))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)
        alpha = 0.06
        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < alpha else 0, axis = 1)
        cv_scores_df[["param_C",	"param_epsilon",	"param_gamma",	"params", "mean_train_score", "mean_test_score", "keep"]].to_csv('0.06_cv_scores_svr.csv')
        #which hyper parameters to filter ?
        C_to_filter_df = cv_scores_df.groupby(["param_C"], as_index = False)["keep"].sum()
        C_to_filter_df["param_C"] = C_to_filter_df.apply(lambda x : 1 if x.keep == 0 else 0, axis=1)
        print(C_to_filter_df)
        gamma_to_filter_df = cv_scores_df.groupby(["param_gamma"], as_index = False)["keep"].sum()
        gamma_to_filter_df["param_gamma"] = gamma_to_filter_df.apply(lambda x : 1 if x.keep == 0 else 0, axis=1)
        print(gamma_to_filter_df)
        epsilon_to_filter_df = cv_scores_df.groupby(["param_epsilon"], as_index = False)["keep"].sum()
        epsilon_to_filter_df["param_epsilon"] = epsilon_to_filter_df.apply(lambda x : 1 if x.keep == 0 else 0, axis=1)
        print(epsilon_to_filter_df)
        #kernel_to_filter_df = cv_scores_df.groupby(["param_kernel"], as_index = False)["keep"].sum()
        #kernel_to_filter_df["param_kernel"] = kernel_to_filter_df.apply(lambda x : 1 if x.keep == 0 else 0, axis=1)
        #print(kernel_to_filter_df)
        cv_scores_df = cv_scores_df.loc[cv_scores_df['keep'] == 1]
        if len(cv_scores_df.index) == 0:
            print("In cv, we could not find hyper parameters for which rmse validation and rmse training are close enough" )
            return 0, 0, 0, 0, 0
        else: 
            best = cv_scores_df.loc[cv_scores_df["mean_test_score"].idxmax()]
            RMSE_train = best["mean_train_score"]
            RMSE_validation = best["mean_test_score"]
            best_c = best['param_C']
            best_gamma = best['param_gamma']
            best_eps = best['param_epsilon']
            #best_kernel = best['param_kernel']
            svr_estimator = svm.SVR(C = best_c, gamma = best_gamma, epsilon = best_eps)
            svr_estimator.fit(X_des, np.ravel(Y_des))
            data_predict_train = svr_estimator.predict(X_des)
            return svr_estimator, [best_c, best_gamma, best_eps], data_predict_train, -RMSE_train, -RMSE_validation

    def test_SVR(self, X_des, Y_des, best_parameters):
        model = svm.SVR(C = best_parameters[0], gamma = best_parameters[1], epsilon = best_parameters[2])
        model.fit(X_des, np.ravel(Y_des))
        data_predict_test = model.predict(X_des)    
        return data_predict_test