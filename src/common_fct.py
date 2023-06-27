import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import datetime

##pre-processing
def print_nan_per(df):
    nb_rows = len(df.index)
    for col in df.columns:
        per_missing = (df[col].isna().sum())*100/nb_rows
        print(str(col) + '  :  '+ str(round(per_missing))+' % of missing values')

def remove_nan(df): 
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
def destandardize(Y_true_std, Y_prediction_std, scaler, nb_col):
    #std stands for standardized
    #compute the unscaled rmse
    Y_true_expanded = pd.DataFrame(np.zeros(shape=(len(Y_true_std), nb_col)))
    Y_true_expanded.iloc[:, 0] = Y_true_std.values.flatten()
    Y_true_destd = pd.DataFrame(scaler.inverse_transform(Y_true_expanded))[0] #3 lines above are not be needed. Just need to take the original BC column. 

    Y_prediction_expanded = pd.DataFrame(np.zeros(shape=(len(Y_prediction_std), nb_col)))
    Y_prediction_expanded.iloc[:, 0] = Y_prediction_std[:]
    Y_prediction_destd = pd.DataFrame(scaler.inverse_transform(Y_prediction_expanded))[0]
    return Y_true_destd, Y_prediction_destd

##plot true values and prediction
def BC_plot(Y_true, Y_prediction, datetime, scaler, sub_set):
    #merge datetime_df and unscaled_test_Y based on the index. 
    """Y_true = pd.DataFrame(Y_true).join(datetime)
    Y_true['datetime'] = pd.to_datetime(Y_true['datetime'])
    Y_true.set_index('datetime', inplace=True)

    Y_prediction = pd.DataFrame(Y_prediction).join(datetime)
    Y_prediction['datetime'] = pd.to_datetime(Y_prediction['datetime'])
    Y_prediction.set_index('datetime', inplace=True)"""
    fig, ax = plt.subplots()
    ax.scatter(Y_true.index, Y_true[0], label='True values', color='blue', marker='.')
    ax.scatter(Y_prediction.index, Y_prediction[0], label='Predicted values', color='red', marker='.')
    ax.set_xlabel('Time')
    ax.set_ylabel('BlackCarbon in µg/m3')
    ax.set_title('Actual vs Predicted in the ' + str(sub_set))
    ax.legend()
    plt.show()

def split(df):
    winter_df = df[(df['date'].dt.month >= 12) | (df['date'].dt.month <= 2)]
    pre_monsoon_df = df[(df['date'].dt.month >= 3) & (df['date'].dt.month <= 5)]
    summer_df = df[(df['date'].dt.month >= 6) & (df['date'].dt.month <= 8)]
    post_monsoon_df = df[(df['date'].dt.month >= 9) & (df['date'].dt.month <= 10)]
    return winter_df, pre_monsoon_df, summer_df, post_monsoon_df

def sample_split(start_date, end_date, df):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return filtered_df

def filter_df(start_date, end_date, df):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = df[(df['date'] < start_date) | (df['date'] > end_date)]
    return filtered_df

def year_by_year_plot(df, year, ax):
    one_year_df = df[(df['datetime'].dt.year == year)]
    one_year_df.set_index('datetime', inplace=True)
    ax.scatter(one_year_df.index,one_year_df["BC"], marker='.')
    ax.set_xlim([datetime.date(year, 1, 1), datetime.date(year, 12, 31)])
    ax.axvline(pd.to_datetime(str(year)+'--01'), color='r', linestyle='--', lw=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('BlackCarbon in µg/m3')
    ax.set_title(str(year))

def one_year_plot_2(df, start_year, start_month, end_year, end_month, ax, start_day=1, end_day=1):
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

def BC_plot(path):
    #path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
    df = pd.read_excel(path)
    df['date'] = pd.to_datetime(df['date'])
    df['Hrs.'] = df['Hrs.'].astype(str)
    df['datetime'] = df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['Hrs.']
    df['datetime'] = pd.to_datetime(df['datetime'])
    plt.scatter(df['datetime'], df['BC'], marker='.')
    plt.xlabel('Datetime')
    plt.ylabel('BC')
    plt.title('BC values over time')
    plt.xticks(rotation=45)
    plt.show()
    
if __name__ == '__main__':
    path = "../data/BC_Exposure_data_set_for_Delhi_2018-2019.xlsx"
    df = pd.read_excel(path)
    df['datetime'] = df['date'].astype(str) + ' ' + df['Hrs.'].astype(str)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M:%S')
    """fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    one_year_plot(df[['datetime', 'BC']], 2018, axes[0])
    one_year_plot(df[['datetime', 'BC']], 2019, axes[1])
    plt.tight_layout()
    plt.show()"""
    """fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
    one_year_plot_2(df, 2018, 1, 2018, 11, ax1)
    one_year_plot_2(df, 2018, 12, 2019, 11, ax2)
    plt.tight_layout()
    plt.show()"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(8, 12))
    one_year_plot_2(df, 2018, 12, 2018, 12, ax1, 16, 20)
    one_year_plot_2(df, 2019, 3, 2019, 3, ax2, 18, 24)
    one_year_plot_2(df, 2019, 6, 2019, 6, ax3, 20, 24)
    one_year_plot_2(df, 2019, 11, 2019, 11, ax4, 1, 4)
    plt.tight_layout()
    plt.show()