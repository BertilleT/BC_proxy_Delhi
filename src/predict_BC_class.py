import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
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

class predict_BC_lib():
    #per is the percentage of difference between training and validation scores we are willing to accept
    per = 0.06
    no_param_found = "In cv, we could not find hyper parameters for which the difference between error validation and error training is inferior to alpha = "
    
    # ignore mlp warnings (from the pipeline)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

    def print_nan_per(self, df):
        nb_rows = len(df.index)
        for col in df.columns:
            per_missing = (df[col].isna().sum())*100/nb_rows
            print(str(col) + '  :  '+ str(round(per_missing))+' % of missing values')
    
    def concat_date_time(self, df):
        #store date time in separate df in order to plot later the predictions and true values according to datetime
        df['datetime'] = df['date'].astype(str) + ' ' + df['Hrs.'].astype(str)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        datetime_df = df[["datetime"]]
        #remove the date time columns
        return df, datetime_df

    def impute_RH(self, df, rh):
        rh['From Date'] = pd.to_datetime(rh['From Date'], format='%d-%m-%Y %H:%M')
        rh['date'] = rh['From Date'].dt.date
        rh['date'] = pd.to_datetime(rh['date'], format='%Y-%m-%d')
        rh = rh.drop(['From Date'], axis = 1)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        rh = rh.rename(columns={'RH': 'RH_new'})
        df = df.merge(rh, on='date') 
        df['RH'] = df.apply(lambda x: x.RH_new if np.isnan(x.RH) else x.RH, axis = 1)
        df['RH'] = pd.to_numeric(df['RH'], errors='coerce')
        df = df.drop('RH_new', axis = 1)
        return df

    def concat_SR(self, df, sr):
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        sr['date'] = sr['date'].str.strip()
        sr['date'] = pd.to_datetime(sr['date'], format='%d/%m/%Y')
        df = df.merge(sr, on='date') 
        df['SR'] = pd.to_numeric(df['SR'], errors='coerce')
        return df

    def remove_nan_columns(self, df, RH_included): 
        #1 Remove columns
        columns_to_remove = ["WD", "WS", "Temp", "RF"]
        if RH_included == False: 
            columns_to_remove.append("RH")
        df = df.drop(columns_to_remove, axis = 1)
        #print("We removed the columns: " + str(columns_to_remove))
        return df

    def remove_nan_rows(self, df): 
        #2 Drop null rows
        nb_rows_original = len(df.index)
        df = df.dropna()
        nb_rows_without_nan = len(df.index)
        per_dropped = ((nb_rows_original - nb_rows_without_nan)*100)/nb_rows_original
        print("We dropped: "+str(round(per_dropped))+"% of the original df.")
        print("The df under study contains now " + str(len(df.index)) + " rows. ")
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        return df 

    def split(self, df):
        winter_df = df[(df['date'].dt.month >= 12) | (df['date'].dt.month <= 2)]
        pre_monsoon_df = df[(df['date'].dt.month >= 3) & (df['date'].dt.month <= 5)]
        summer_df = df[(df['date'].dt.month >= 6) & (df['date'].dt.month <= 8)]
        post_monsoon_df = df[(df['date'].dt.month >= 9) & (df['date'].dt.month <= 10)]
        return winter_df, pre_monsoon_df, summer_df, post_monsoon_df

    def filter_df(self, start_date, end_date, df):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = df[(df['date'] < start_date) | (df['date'] > end_date)]
        return filtered_df

    def plot_RH(self, df, rh):
        rh['From Date'] = pd.to_datetime(rh['From Date'], format='%d-%m-%Y %H:%M')
        rh['date'] = rh['From Date'].dt.date
        rh['date'] = pd.to_datetime(rh['date'], format='%Y-%m-%d')
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        rh = rh.rename(columns={'RH': 'RH_new'})
        df = df.merge(rh, on='date') 
        df['RH_new'] = df['RH_new'].replace('None', np.nan).astype(float)
        #df = df.groupby(pd.Grouper(key='date', axis=0, freq='d')).mean()
        df = df.reset_index()
        fig, ax = plt.subplots()

        ax.plot(df['date'], df['RH'], label='RH', color='b', lw=2)
        ax.plot(df['date'], df['RH_new'], label='RH_new', color="r")
        ax.set_xlabel('Time')
        ax.set_ylabel('RH')
        ax.legend()
        plt.show()
    
    ##plot true values and prediction according to time
    def trueANDpred_time_plot(self, Y_true, Y_prediction, datetime, method, season, save_images):
        #merge datetime_df and unscaled_test_Y based on the index. 
        Y_true = pd.DataFrame(Y_true).join(datetime)
        Y_true['datetime'] = pd.to_datetime(Y_true['datetime'])
        Y_true.set_index('datetime', inplace=True)

        Y_prediction = pd.DataFrame(Y_prediction).join(datetime)
        Y_prediction['datetime'] = pd.to_datetime(Y_prediction['datetime'])
        Y_prediction.set_index('datetime', inplace=True)
        Y_true = Y_true.sort_index()
        Y_prediction = Y_prediction.sort_index()
        if season == 'winter': #and RH_included == True and RH_imputed == False:
            nb_rows = 144 #6 days (there are missing values between the 6th and 7th day)
        else:
            nb_rows = 168 # = 24*7 = one week
        Y_true = Y_true.head(nb_rows)
        Y_prediction = Y_prediction.head(nb_rows)

        fig, ax = plt.subplots(figsize=(12,9))
        ax.plot(Y_true.index, Y_true["BC"], label='True values', color='blue')
        ax.plot(Y_prediction.index, Y_prediction["BC"], label='Predicted values', color='red',)# marker='.')
        ax.set_xlim(Y_true.index.min(), Y_true.index.max())
        ax.set_xlabel('Time')
        ax.set_ylabel('BlackCarbon in µg/m3')
        ax.set_title('Testing: true vs predicted values with ' + method + ' in ' + str(season))
        ax.legend()
        if save_images == True:
            fig.savefig('../img/' + method + '/predictedANDtrue_'+ method +'_' + season + '.png')

    ##plot scatter of true values against prediction 
    def trueVSpred_scatter_plot(self, Y_true, Y_prediction, method, season, save_images):
        fig, ax = plt.subplots(figsize=(9,9))
        ax.scatter(Y_true.values, Y_prediction.values)
        ax.plot([min(Y_true["BC"]), max(Y_true["BC"])], [min(Y_true["BC"]), max(Y_true["BC"])], color='red', linestyle='--')
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0) 
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Testing: predicted values with ' + method + ' vs true values in ' + str(season))
        if save_images == True:
            fig.savefig('../img/' + method + '/predictedVStrue_'+ method +'_' + season + '.png')

    def one_year_plot(self, df, start_year, start_month, end_year, end_month, ax, start_day=1, end_day=1):
        start_date = pd.to_datetime(f"{start_year}-{start_month}-{start_day}", format='%Y-%m-%d')
        end_date = pd.to_datetime(f"{end_year}-{end_month}-{end_day}", format='%Y-%m-%d') + pd.offsets.MonthEnd(1)
        one_year_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        one_year_df.set_index('datetime', inplace=True)
        ax.scatter(one_year_df.index, one_year_df["BC"], marker='.')
        ax.set_xlim([datetime.date(end_year-1, 12, 1), datetime.date(end_year, end_month, 30)])

        if plt.gca() is ax:
            ax.axvline(pd.to_datetime('2018-12-01'), color='r', linestyle='--', lw=2)
            ax.axvline(pd.to_datetime('2018-03-01'), color='g', linestyle='--', lw=2)
            ax.axvline(pd.to_datetime('2018-06-01'), color='purple', linestyle='--', lw=2)
            ax.axvline(pd.to_datetime('2018-09-01'), color='b', linestyle='--', lw=2)
            ax.text(datetime.date(2018, 1, 10), 110, "winter", color = "r", fontsize = 17)
            ax.text(datetime.date(2018, 4, 1), 110, "pre-monsoon", color = "g", fontsize = 17)
            ax.text(datetime.date(2018, 7, 10), 110, "monsoon", color = "purple", fontsize = 17)
            ax.text(datetime.date(2018, 10, 1), 110, "post-monsoon", color = "b", fontsize = 17)
        else: 
            ax.axvline(pd.to_datetime('2019-12-01'), color='r', linestyle='--', lw=2)
            ax.axvline(pd.to_datetime('2019-03-01'), color='g', linestyle='--', lw=2)
            ax.axvline(pd.to_datetime('2019-06-01'), color='purple', linestyle='--', lw=2)
            ax.axvline(pd.to_datetime('2019-09-01'), color='b', linestyle='--', lw=2)
            ax.text(datetime.date(2019, 1, 10), 110, "winter", color = "r", fontsize = 17)
            ax.text(datetime.date(2019, 4, 1), 110, "pre-monsoon", color = "g", fontsize = 17)
            ax.text(datetime.date(2019, 7, 10), 110, "monsoon", color = "purple", fontsize = 17)
            ax.text(datetime.date(2019, 10, 1), 110, "post-monsoon", color = "b", fontsize = 17)
        ax.set_xlabel('Time')
        ax.set_ylabel('BlackCarbon in µg/m3')
        ax.set_title(f'{start_date:%b %Y} - {end_date:%b %Y}')
        

    def season_split_plot(self, df, RH_included, RH_imputed):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
        plt.sca(ax1)
        self.one_year_plot(df, 2018, 1, 2018, 11, ax1)
        self.one_year_plot(df, 2018, 12, 2019, 11, ax2)
        plt.tight_layout()
        plt.show()
        """if save_images == True:
            if RH_included == True:
                if RH_imputed == True: 
                    fig.savefig('../img/seasons_split/BC_seasons_RH_imputed.png')
                elif RH_imputed == False: 
                    fig.savefig('../img/seasons_split/BC_seasons_RH_included_nan_dropped.png')
            elif RH_included == False:
                fig.savefig('../img/seasons_split/BC_seasons_RH_excluded.png')"""

    def train_RF(self, X, Y, scoring, best_params, std_all_training):   
        alpha = predict_BC_lib.per * (Y.quantile(0.9))
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
       
        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < alpha else 0, axis = 1) 
        cv_scores_df = cv_scores_df.loc[cv_scores_df['keep'] == 1]    
        
        if len(cv_scores_df.index) == 0:
            print(predict_BC_lib.no_param_found + str(alpha))
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

    def train_SVR(self, X, Y, scoring, best_params, std_all_training):
        alpha = predict_BC_lib.per * (Y.quantile(0.9))
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
            print(predict_BC_lib.no_param_found + str(alpha))
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
    
    def train_NN(self, X, Y, scoring, best_params, std_all_training): 
        alpha = predict_BC_lib.per * (Y.quantile(0.9))
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
            print(predict_BC_lib.no_param_found + str(alpha))
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
