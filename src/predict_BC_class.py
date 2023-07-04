import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV

class predict_BC_lib():
    def print_nan_per(self, df):
        nb_rows = len(df.index)
        for col in df.columns:
            per_missing = (df[col].isna().sum())*100/nb_rows
            print(str(col) + '  :  '+ str(round(per_missing))+' % of missing values')

    def remove_nan(self, df): 
        nb_rows_original = len(df.index)
        columns_to_remove = ["WD", "WS", "Temp", "RF"]
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
        #compute the unscaled rmse
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

    ##plot true values and prediction
    def BC_plot(self, Y_true, Y_prediction, datetime, scaler, season):
        #merge datetime_df and unscaled_test_Y based on the index. 
        Y_true = pd.DataFrame(Y_true).join(datetime)
        Y_true['datetime'] = pd.to_datetime(Y_true['datetime'])
        Y_true.set_index('datetime', inplace=True)

        Y_prediction = pd.DataFrame(Y_prediction).join(datetime)
        Y_prediction['datetime'] = pd.to_datetime(Y_prediction['datetime'])
        Y_prediction.set_index('datetime', inplace=True)
        print("Y_true", Y_true)
        print("Y_prediction", Y_prediction)
        Y_true = Y_true.sort_index()
        Y_prediction = Y_prediction.sort_index()
        Y_true = Y_true.head(168)
        Y_prediction = Y_prediction.head(168)
        print(Y_true)
        fig, ax = plt.subplots()
        ax.plot(Y_true.index, Y_true[0], label='True values', color='blue')
        ax.plot(Y_prediction.index, Y_prediction[0], label='Predicted values', color='red',)# marker='.')
        ax.set_xlabel('Time')
        ax.set_ylabel('BlackCarbon in µg/m3')
        ax.set_title('Actual vs Predicted in the ' + str(season))
        ax.legend()
        plt.show()

    def split(self, df):
        winter_df = df[(df['date'].dt.month >= 12) | (df['date'].dt.month <= 2)]
        pre_monsoon_df = df[(df['date'].dt.month >= 3) & (df['date'].dt.month <= 5)]
        summer_df = df[(df['date'].dt.month >= 6) & (df['date'].dt.month <= 8)]
        post_monsoon_df = df[(df['date'].dt.month >= 9) & (df['date'].dt.month <= 10)]
        return winter_df, pre_monsoon_df, summer_df, post_monsoon_df

    def sample_split(self, start_date, end_date, df):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        return filtered_df

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

    def one_year_plot_2(self, df, start_year, start_month, end_year, end_month, ax, start_day=1, end_day=1):
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

    def train_RF(self, X_des, Y_des, scoring):        
        kfold = 10
        n_estimators = [500]
        max_features = [20]
        max_depth = [3]
        param_grid = { 'n_estimators' : n_estimators, 'max_features': max_features, 'max_depth': max_depth}
        rf_estimator = RandomForestRegressor()
        search = GridSearchCV(rf_estimator, scoring=scoring, param_grid = param_grid, cv = kfold, refit = False, return_train_score=True)
        search.fit(X_des, np.ravel(Y_des))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)
        #print(cv_scores_df[["mean_train_score", "mean_test_score"]].to_string())
        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < 0.06 else 0, axis = 1)
        #cv_scores_df[["param_n_estimators",	"param_max_features",	"param_max_depth",	"params", "mean_train_score", "mean_test_score", "keep"]].to_csv('scores/0.06_all_cv_scores_rf_without_PM10.csv')
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

    def train_SVR(self, X_des, Y_des, metric):
        kfold = 10
        cs = [10]
        gs = [0.1]
        epsilons = [0.1]
        #kernels = ["rbf", "poly", "sigmoid"]
        param_grid = { 'C' : cs, 'gamma': gs, 'epsilon': epsilons}#, 'kernel': kernels}
        svr_estimator = svm.SVR()
        search = GridSearchCV(svr_estimator, scoring = 'neg_mean_absolute_error', param_grid = param_grid, cv = kfold, refit = False, return_train_score=True)
        search.fit(X_des, np.ravel(Y_des))
        cv_scores_df = pd.DataFrame.from_dict(search.cv_results_)
        alpha = 1#0.06
        cv_scores_df["keep"] = cv_scores_df.apply(lambda x: 1 if np.absolute(x.mean_train_score - x.mean_test_score) < alpha else 0, axis = 1)
        #cv_scores_df[["param_C",	"param_epsilon",	"param_gamma",	"params", "mean_train_score", "mean_test_score", "keep"]].to_csv('0.06_cv_scores_svr.csv')
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
        """kernel_to_filter_df = cv_scores_df.groupby(["param_kernel"], as_index = False)["keep"].sum()
        kernel_to_filter_df["param_kernel"] = kernel_to_filter_df.apply(lambda x : 1 if x.keep == 0 else 0, axis=1)
        print(kernel_to_filter_df)"""
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
    