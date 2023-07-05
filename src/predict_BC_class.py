import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class predict_BC_lib():
    alpha = 0.8
    no_param_found = "In cv, we could not find hyper parameters for which the difference between error validation and error training is inferior to alpha = "
    def print_nan_per(self, df):
        nb_rows = len(df.index)
        for col in df.columns:
            per_missing = (df[col].isna().sum())*100/nb_rows
            print(str(col) + '  :  '+ str(round(per_missing))+' % of missing values')

    def remove_nan(self, df): 
        nb_rows_original = len(df.index)
        columns_to_remove = ["WD", "WS", "Temp", "RF", "RH"]
        df = df.drop(columns_to_remove, axis = 1)
        #print("We removed the columns: " + str(columns_to_remove))
        df = df.dropna()
        nb_rows_without_nan = len(df.index)
        per_dropped = ((nb_rows_original - nb_rows_without_nan)*100)/nb_rows_original
        print("We dropped: "+str(round(per_dropped))+"% of the original df.")
        print("The df under study contains now " + str(len(df.index)) + " rows. ")
        return df 

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
        #plt.show()
        ax.legend()
        fig.savefig('../img/' + method + '_without_RH/predictedANDtrue_'+ method +'_' + season + '.png')

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
        #plt.show()
        fig.savefig('../img/' + method + '_without_RH/predictedVStrue_'+ method +'_' + season + '.png')

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

    def year_by_year_plot(self, df, year, ax):
        one_year_df = df[(df['datetime'].dt.year == year)]
        one_year_df.set_index('datetime', inplace=True)
        ax.scatter(one_year_df.index,one_year_df["BC"], marker='.')
        ax.set_xlim([datetime.date(year, 1, 1), datetime.date(year, 12, 31)])
        ax.axvline(pd.to_datetime(str(year)+'--01'), color='r', linestyle='--', lw=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('BlackCarbon in µg/m3')
        ax.set_title(str(year))

    def one_year_plot(self, df, start_year, start_month, end_year, end_month, ax, start_day=1, end_day=1):
        start_date = pd.to_datetime(f"{start_year}-{start_month}-{start_day}", format='%Y-%m-%d')
        end_date = pd.to_datetime(f"{end_year}-{end_month}-{end_day}", format='%Y-%m-%d') #+ pd.offsets.MonthEnd(1)
        one_year_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        one_year_df.set_index('datetime', inplace=True)
        ax.scatter(one_year_df.index, one_year_df["BC"], marker='.')
        """ax.set_xlim([datetime.date(end_year-1, 12, 1), datetime.date(end_year, end_month, 30)])
        ax.axvline(pd.to_datetime('2018-12-01'), color='r', linestyle='--', lw=2)
        ax.axvline(pd.to_datetime('2018-03-01'), color='g', linestyle='--', lw=2)
        ax.axvline(pd.to_datetime('2018-06-01'), color='purple', linestyle='--', lw=2)
        ax.axvline(pd.to_datetime('2018-09-01'), color='b', linestyle='--', lw=2)


        ax.axvline(pd.to_datetime('2019-12-01'), color='r', linestyle='--', lw=2)
        ax.axvline(pd.to_datetime('2019-03-01'), color='g', linestyle='--', lw=2)
        ax.axvline(pd.to_datetime('2019-06-01'), color='purple', linestyle='--', lw=2)
        ax.axvline(pd.to_datetime('2019-09-01'), color='b', linestyle='--', lw=2)"""
        ax.set_xlabel('Time')
        ax.set_ylabel('BlackCarbon in µg/m3')
        ax.set_title(f'{start_date:%b %Y} - {end_date:%b %Y}')

    def train_RF(self, X, Y, scoring, best_params):        
        kfold = 10
        if best_params != 'null':
            n_estimators = [best_params[0]]
            max_features = [best_params[1]]
            max_depth = [best_params[2]]
        else: 
            n_estimators = [50, 100, 300, 500, 1000]
            max_features = [1, 3, 5, 10, 15]
            max_depth = [5, 10, 15, 20, 30]
        param_grid = { 'n_estimators' : n_estimators, 'max_features': max_features, 'max_depth': max_depth}
        rf_estimator = RandomForestRegressor()
        search = GridSearchCV(rf_estimator, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True)
        search.fit(X, np.ravel(Y))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)
        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < predict_BC_lib.alpha else 0, axis = 1)        
        if len(cv_scores_df.index) == 0:
            print(predict_BC_lib.no_param_found + str(predict_BC_lib.alpha))
            return 0, 0, 0, 0, 0
        else: 
            best = cv_scores_df.loc[cv_scores_df["mean_test_score"].idxmax()]
            error_train = best["mean_train_score"]
            error_validation = best["mean_test_score"]

            best_n = best['param_n_estimators']#500
            best_features = best['param_max_features']#10
            best_depth = best['param_max_depth']#5
            rf_estimator = RandomForestRegressor(n_estimators = best_n, max_features = best_features, max_depth = best_depth)
            rf_estimator.fit(X, np.ravel(Y))
            data_predict_train = rf_estimator.predict(X)
            return rf_estimator, [best_n, best_features, best_depth], data_predict_train, -error_train, -error_validation

    def train_SVR(self, X, Y, scoring, best_params):
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
        search = GridSearchCV(svr_estimator, scoring = scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True)
        search.fit(X, np.ravel(Y))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)
        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < predict_BC_lib.alpha else 0, axis = 1)
        cv_scores_df = cv_scores_df.loc[cv_scores_df['keep'] == 1]
        if len(cv_scores_df.index) == 0:
            print(predict_BC_lib.no_param_found + str(predict_BC_lib.alpha))
            return 0, 0, 0, 0, 0
        else: 
            best = cv_scores_df.loc[cv_scores_df["mean_test_score"].idxmax()]
            error_train = best["mean_train_score"]
            error_validation = best["mean_test_score"]
            best_c = best['param_C']
            best_gamma = best['param_gamma']
            best_eps = best['param_epsilon']
            #best_kernel = best['param_kernel']
            svr_estimator = svm.SVR(C = best_c, gamma = best_gamma, epsilon = best_eps)
            svr_estimator.fit(X, np.ravel(Y))
            data_predict_train = svr_estimator.predict(X)
            return svr_estimator, [best_c, best_gamma, best_eps], data_predict_train, -error_train, -error_validation
    
    def train_NN(self, X_train, Y_train, X_validation, Y_validation, scoring, best_params): 
        if best_params != 'null':
            num_layers = best_params[0]
            learning_rate = best_params[1]
            nb_epochs = best_params[2]
            hidden_size = best_params[3]
        else: 
            num_layers = [1, 2, 3, 4]
            learning_rate = [0.001, 0.01, 0.1]
            nb_epochs = [10, 30, 50, 100]
            hidden_size = []
        train = data_utils.TensorDataset(torch.Tensor(np.array(X_train)), torch.Tensor(np.array(Y_train)))
        train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
        torch.manual_seed(42)

        mlp = nn.Sequential(nn.Linear(X_train.shape[1], hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

        loss_function = nn.MSELoss(size_average=False)
        SGD_optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
        for epoch in range(nb_epochs): 
            print(f'Starting epoch {epoch+1}')
            current_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))
                SGD_optimizer.zero_grad()
                outputs = mlp(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                SGD_optimizer.step()        
        with torch.no_grad():
            train_predicted_Y = mlp(torch.Tensor(np.array(X_train)))
            valid_predicted_Y = mlp(torch.Tensor(np.array(X_validation)))

        train_predicted_Y_np = train_predicted_Y.detach().numpy()
        valid_predicted_Y_np = valid_predicted_Y.detach().numpy()
        
        if scoring == 'neg_mean_squared_error':
            error_train = np.sqrt(mean_squared_error(Y_train, train_predicted_Y))
            error_validation = np.sqrt(mean_squared_error(Y_validation, valid_predicted_Y))
        elif scoring == 'neg_mean_absolute_error':
            error_train = np.sqrt(mean_absolute_error(Y_train, train_predicted_Y))
            error_validation = mean_absolute_error(Y_validation, valid_predicted_Y)

        return  mlp, best_params, train_predicted_Y_np, error_train, error_validation