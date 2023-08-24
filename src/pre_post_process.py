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

class Pre_post_process():
    # Print the percentage of nan values for each column of df
    def print_nan_per(self, df):
        nb_rows = len(df.index)
        for col in df.columns:
            per_missing = (df[col].isna().sum())*100/nb_rows
            print(str(col) + '  :  '+ str(round(per_missing))+' % of missing values')
    
    # Concatenate date and time columns into one new column datetime
    def concat_date_time(self, df):
        #store date time in separate df in order to plot later the predictions and true values according to datetime
        df['datetime'] = df['date'].astype(str) + ' ' + df['Hrs.'].astype(str)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        datetime_df = df[["datetime"]]
        #remove the date time columns
        return df, datetime_df

    # When the Relative Humidty measures are missing in df dataframe, impute them with the values from rh dataframe. 
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

    # Add the Solar Radiation column from sr dataframe, to df dataframe. 
    def concat_SR(self, df, sr):
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        sr['date'] = sr['date'].str.strip()
        sr['date'] = pd.to_datetime(sr['date'], format='%d/%m/%Y')
        df = df.merge(sr, on='date') 
        df['SR'] = pd.to_numeric(df['SR'], errors='coerce')
        return df

    # Remove columns with 99 or 100% of nan values. 
    def remove_nan_columns(self, df, RH_included): 
        #1 Remove columns
        columns_to_remove = ["WD", "WS", "Temp", "RF"]
        if RH_included == False: 
            columns_to_remove.append("RH")
        df = df.drop(columns_to_remove, axis = 1)
        #print("We removed the columns: " + str(columns_to_remove))
        return df

    # For each col in df, remove rows with nan values
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

    # Split df into 4 seasonal subsets df 
    def split(self, df):
        winter_df = df[(df['date'].dt.month >= 12) | (df['date'].dt.month <= 2)]
        pre_monsoon_df = df[(df['date'].dt.month >= 3) & (df['date'].dt.month <= 5)]
        summer_df = df[(df['date'].dt.month >= 6) & (df['date'].dt.month <= 8)]
        post_monsoon_df = df[(df['date'].dt.month >= 9) & (df['date'].dt.month <= 10)]
        return winter_df, pre_monsoon_df, summer_df, post_monsoon_df

    # Select the rows from df with datetime between start_date and end_date
    def filter_df(self, start_date, end_date, df):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = df[(df['date'] < start_date) | (df['date'] > end_date)]
        return filtered_df