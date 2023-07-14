import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import datetime

class Plot():
    def RH(self, df, rh):
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
    def trueANDpred_time(self, Y_true, Y_prediction, datetime, method, season, save_images):
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
    def trueVSpred_scatter(self, Y_true, Y_prediction, method, season, save_images):
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

    def one_year(self, df, start_year, start_month, end_year, end_month, ax, start_day=1, end_day=1):
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
        

    def season_split(self, df, RH_included, RH_imputed):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
        plt.sca(ax1)
        self.one_year(df, 2018, 1, 2018, 11, ax1)
        self.one_year(df, 2018, 12, 2019, 11, ax2)
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