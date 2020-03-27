#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:45:21 2020

@author: maxenceleclercq
"""

import pandas as pd 
import numpy as np 
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def influent_dates(x):
    tab = []
    for i in x:
        tab.append(i)
    return tab

class PreProcessor():
    def __init__(self, file):
        self.file = file
        self.scaler = MinMaxScaler()
        
        self.df_raw = None
        self.df = None
        self.df_train = None
        self.df_test = None
        
    def data_importation(self):
        self.df_raw = pd.read_csv(self.file)
        
    def model_processing(self, N_t, threshold):
        
        df_raw = self.dataframe_processing()
                    
        df_diff = self.new_cases(df_raw)
        
        df_1 = self.columns_selection(df_raw, 'Cases')
        
        df_2 = self.columns_selection(df_diff, 'New Cases')
        
        df = pd.merge(df_1, df_2, on=['date', 'Zone'], how='left')
        
        df = self.scaling(df)
    
        df = self.last_cases(df, N_t)
        
        self.df = df.copy()
        
        self.df_train = self.df[self.df['date'] <= pd.Timestamp(threshold)]
        self.df_test = self.df[self.df['date'] > pd.Timestamp(threshold)]
        
    def dataframe_processing(self):
        df = self.df_raw.transpose()
        df.columns = df.iloc[1, :].astype(str) + '_' +  df.iloc[0, :].astype(str)
        df = df.iloc[4:]
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['index'])
        df.columns = df.columns.str.replace(" ","_")
        df.columns = df.columns.str.replace(",","_")
        df = df.drop(['index', ], axis=1)
        df = df.set_index('date')
        
        return df
    
    def new_cases(self, df_raw):
        df_diff = df_raw.copy()
        for i in range(1, len(df_diff)):
            df_diff.loc[df_raw.index.min() + dt.timedelta(days=len(df_raw) - i)] = df_diff.loc[df_raw.index.min() + dt.timedelta(days=len(df_raw) - i)] - \
                        df_diff.loc[df_raw.index.min() + dt.timedelta(days=len(df_raw) - i - 1)]
                        
        return df_diff
    
    def scaling(self, df):
        
        columns_to_scale = list(set(df.columns) - set(['date', 'Zone']))
        
        df[columns_to_scale] = pd.DataFrame(self.scaler.fit_transform(df[columns_to_scale]), 
          columns=columns_to_scale, index=df.index)
        
        return df
        
    def columns_selection(self, df_raw, name_col):
        columns_china = []
        for column in df_raw.columns:
            if column[:14] == 'Mainland_China':
                columns_china.append(column)
        # columns = columns_china + \
        #             ['South_Korea_nan', 'Italy_nan', 'Hong_Kong_Hong_Kong']
        
        # columns = columns_china
        
        columns=['Mainland_China_Hubei']
                    
        df = df_raw[columns]
        
        df = pd.DataFrame(df.stack().reset_index().rename(columns={'level_0': 'date', 
                                                                   'level_1':'Zone',
                                                                   0: name_col}))
        
        return df
        
    def last_cases(self, df_raw, N_t):
        df_merge = pd.merge(df_raw, df_raw, on='Zone')
        df_merge = df_merge[(df_merge['date_x'] > df_merge['date_y']) & (df_merge['date_x'] <= 
                                                                 df_merge['date_y'] + dt.timedelta(days=N_t))].copy()
        
        df = pd.DataFrame()
        grouper = df_merge.groupby(['date_x', 'Zone'])
        
        df['Count'] = grouper.count()['Cases_x']
        
        features = list(set(df_raw.columns) - set(['date', 'Zone']))
        for feature in features:
            df[feature] = grouper.max()[feature + '_x']
            df['Last ' + feature] = grouper[feature + '_y'].apply(influent_dates)
        
        df = df[df['Count'] == N_t]
        df = df.drop(columns=['Count'])
        
        df.index = df.index.rename(['date', 'Zone'])
        df = df.reset_index()
        
        return df
        