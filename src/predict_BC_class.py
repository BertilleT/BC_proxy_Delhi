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
    
    ##destandardize the prediction
    def destandardize(self, Y_true_std, Y_prediction_std, scaler, nb_col):
        #std stands for standardized
        #compute the unscaled error
        Y_true_expanded = pd.DataFrame(np.zeros(shape=(len(Y_true_std), nb_col)))
        #Y_true_expanded.iloc[:, 0] = Y_true_std.values.flatten()
        Y_true_index = Y_true_std.index
        Y_true_expanded.iloc[:, 0] = Y_true_std[["BC"]].to_numpy()
        Y_true_destd = pd.DataFrame(scaler.inverse_transform(Y_true_expanded))[0]

        Y_prediction_expanded = pd.DataFrame(np.zeros(shape=(len(Y_prediction_std), nb_col)))
        Y_prediction_expanded.iloc[:, 0] = Y_prediction_std[:]
        Y_prediction_destd = pd.DataFrame(scaler.inverse_transform(Y_prediction_expanded))[0]
        
        Y_true_destd = Y_true_destd.set_axis(Y_true_index)
        Y_prediction_destd = Y_prediction_destd.set_axis(Y_true_index)
        #keep index is important. The predictions do not have them but the true values yes. 
        return Y_true_destd, Y_prediction_destd

    def plot_RH(self, df, rh):
        rh['From Date'] = pd.to_datetime(rh['From Date'], format='%d-%m-%Y %H:%M')
        rh['date'] = rh['From Date'].dt.date
        rh['date'] = pd.to_datetime(rh['date'], format='%Y-%m-%d')
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        rh = rh.rename(columns={'RH': 'RH_new'})
        df = df.merge(rh, on='date') 
        df['RH_new'] = df['RH_new'].replace('None', np.nan).astype(float)
        df = df.groupby(pd.Grouper(key='date', axis=0, freq='d')).mean()
        df = df.reset_index()
        fig, ax = plt.subplots()

        ax.plot(df['date'], df['RH'], label='RH', color='b', lw=2)
        ax.plot(df['date'], df['RH_new'], label='RH_new', color="r")
        ax.set_xlabel('Time')
        ax.set_ylabel('RH')
        ax.legend()
        plt.show()
    
    ##plot true values and prediction according to time
    def trueANDpred_time_plot(self, Y_true, Y_prediction, datetime, method, season):
        #merge datetime_df and unscaled_test_Y based on the index. 
        Y_true = pd.DataFrame(Y_true).join(datetime)
        Y_true['datetime'] = pd.to_datetime(Y_true['datetime'])
        Y_true.set_index('datetime', inplace=True)

        Y_prediction = pd.DataFrame(Y_prediction).join(datetime)
        Y_prediction['datetime'] = pd.to_datetime(Y_prediction['datetime'])
        Y_prediction.set_index('datetime', inplace=True)
        Y_true = Y_true.sort_index()
        Y_prediction = Y_prediction.sort_index()

        Y_true = Y_true.head(168)
        Y_prediction = Y_prediction.head(168)

        fig, ax = plt.subplots(figsize=(12,9))
        ax.plot(Y_true.index, Y_true[0], label='True values', color='blue')
        ax.plot(Y_prediction.index, Y_prediction[0], label='Predicted values', color='red',)# marker='.')
        ax.set_xlim(Y_true.index.min(), Y_true.index.max())
        ax.set_xlabel('Time')
        ax.set_ylabel('BlackCarbon in µg/m3')
        ax.set_title('Testing: true vs predicted values with ' + method + ' in ' + str(season))
        ax.legend()
        plt.show()
        #fig.savefig('../img/' + method + '/predictedANDtrue_'+ method +'_' + season + '.png')

    ##plot scatter of true values against prediction 
    def trueVSpred_scatter_plot(self, Y_true, Y_prediction, method, season):
        fig, ax = plt.subplots(figsize=(9,9))
        ax.scatter(Y_true, Y_prediction)
        ax.plot([min(Y_true), max(Y_true)], [min(Y_true), max(Y_true)], color='red', linestyle='--')
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0) 
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Testing: predicted values with ' + method + ' vs true values in ' + str(season))
        plt.show()
        #fig.savefig('../img/' + method + '/predictedVStrue_'+ method +'_' + season + '.png')

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
        

    def season_split_plot(self, df):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
        plt.sca(ax1)
        self.one_year_plot(df, 2018, 1, 2018, 11, ax1)
        self.one_year_plot(df, 2018, 12, 2019, 11, ax2)
        plt.tight_layout()
        plt.show()

    def train_RF(self, X, Y, scoring, best_params):   
        alpha = predict_BC_lib.per * (Y.quantile(0.9))
        alpha = alpha.item()
        kfold = 10

        if best_params != 'null':
            n_estimators = [best_params[0]]
            max_features = [best_params[1]]
            max_depth = [best_params[2]]
        else: 
            #n_estimators = [50, 100, 300, 500, 1000]
            #max_features = [1, 3, 5, 10, 15]
            #max_depth = [5, 10, 15, 20, 30]
            n_estimators = [500]
            max_features = [1, 3, 10]
            max_depth = [5, 10]

        param_grid = {'model__n_estimators' : n_estimators, 'model__max_features': max_features, 'model__max_depth': max_depth}

        scaler = preprocessing.StandardScaler()
        rf_estimator = RandomForestRegressor()
        pipe = Pipeline([('scaler', scaler), ('model', rf_estimator)])
        search = GridSearchCV(pipe, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True, n_jobs=2)
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
            best_n = best['param_model__n_estimators']#500
            best_features = best['param_model__max_features']#10
            best_depth = best['param_model__max_depth']#5
            rf_estimator = RandomForestRegressor(n_estimators = best_n, max_features = best_features, max_depth = best_depth)
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
    
    def train_NN(self, X, Y, scoring, best_params): 
        #simplefilter(action='ignore', category=FutureWarning)
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
            """nb_neurons = [(50,), (100, 50), (100, 100, 50)]
            activation = ['logistic', 'relu']
            optimizer = ['sgd', 'adam']
            alpha_nn = [0.0001, 0.001, 0.01]
            learning_rate = ['constant', 'adaptive']"""
            nb_neurons = [(50,), (100, 50), (100, 100, 50)]
            activation = ['relu']
            optimizer = ['adam']
            alpha_nn = [0.0001, 0.001, 0.01]
            learning_rate = ['constant']
        
        param_grid = { 'model__hidden_layer_sizes' : nb_neurons, 'model__activation': activation, 'model__solver': optimizer, 'model__alpha': alpha_nn, 'model__learning_rate': learning_rate}
        
        scaler = preprocessing.StandardScaler()
        mlp_estimator = MLPRegressor(random_state=1, max_iter=50)
        pipe = Pipeline([('scaler', scaler), ('model', mlp_estimator)]) 
        search = GridSearchCV(pipe, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True, n_jobs=2)
        #warnings.filterwarnings("ignore")
        #warnings.warn = self.warn

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

            best_hidden_layer_sizes = best['param_model__hidden_layer_sizes']
            best_activation = best['param_model__activation']
            best_solver = best['param_model__solver']
            best_alpha = best['param_model__alpha']
            best_learning_rate = best['param_model__learning_rate']

            mlp_estimator = MLPRegressor(hidden_layer_sizes = best_hidden_layer_sizes, activation = best_activation, solver = best_solver, alpha = best_alpha, learning_rate = best_learning_rate)
            mlp_estimator.fit(X, np.ravel(Y))
            data_predict_train = mlp_estimator.predict(X)
            return mlp_estimator, [best_hidden_layer_sizes, best_activation, best_solver, best_alpha, best_learning_rate], data_predict_train, -error_train, -error_validation
