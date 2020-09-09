import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

class preprocess_data:

    def __init__(self):
        self.le = LabelEncoder()

    def type_of_file(self, file):
        dict_prova = {}
        dict_prova = {'type': file}
        st.write(dict_prova)

    def training(self, filename, filename1, submit ='no', verbose = 1, city = 0, rem_feat = 'no', additional_feat_rem = 'no', adding_feature = 'no', num_weeks = 1):
        df = pd.read_csv(filename)
        df_labels = pd.read_csv(filename1)
        df_conc = pd.merge(df, df_labels, on=['city','year','weekofyear'])

        # df_conc.rename(columns={"city": "city_type"}, inplace = True)
        # df_conc['city']=self.le.fit_transform(df_conc['city_type'])
        # mapping = dict(zip(self.le.classes_, range(len(self.le.classes_))))
        # st.write(mapping)

        if city == 0:
            df_conc = df_conc[df_conc['city'] == 'sj']
        else:
            df_conc = df_conc[df_conc['city'] == 'iq']
        # df_conc.drop(['city_type'], axis = 1, inplace=True)
        df_conc.drop(['city'], axis = 1, inplace=True)

        # df_conc.interpolate(method='linear', limit_direction='forward', axis = 0, inplace=True)
        df_conc.fillna(method='ffill', inplace=True)
        # df_conc.dropna(inplace = True)

        df_conc["week_start_date"] = pd.to_datetime(df_conc['week_start_date'])
        df_conc.set_index('week_start_date', inplace = True)

        weeks = [i for i in range(df_conc.shape[0])]
        df_conc['weeks'] = weeks
        df_conc.drop(['weekofyear', 'year'], axis = 1, inplace=True)

        if verbose == 1:
            st.write(df_conc.T)

        #smoothing funcion
        # df_conc = df_conc.ewm(span = 8).mean()

        feature_to_remove, feats = self.feature_remove(df_conc, verbose, rem_feat, additional_feat_rem)
        # self.equilibrare_i_pesi(df_conc)

        X=df_conc.drop(feature_to_remove, axis=1)
        if adding_feature == 'yes':
            X = self.add_features(X, num_weeks)

        Y=df_conc['total_cases']
        perc = 0.295
        if submit == 'no':
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = perc, random_state = 10)
            df_time = {'train':x_train.index.tolist(), 'test':x_test.index.tolist() }

            x_train = x_train.reset_index(drop = True)
            x_test = x_test.reset_index(drop = True)
            y_train = y_train.reset_index(drop = True)
            y_test = y_test.reset_index(drop = True)
            return x_train, x_test, y_train, y_test, feats, df_time

        elif submit == 'time':
            perc = 1 - perc
            test_size = round(perc*X.shape[0])
            x_train = X.head(test_size)
            x_test = X.tail(X.shape[0] - test_size)

            y_train = Y.head(test_size)
            y_test = Y.tail(Y.shape[0] - test_size)
            df_time = {'train':x_train.index.tolist(), 'test':x_test.index.tolist() }

            x_train = x_train.reset_index(drop = True)
            x_test = x_test.reset_index(drop = True)
            y_train = y_train.reset_index(drop = True)
            y_test = y_test.reset_index(drop = True)
            return x_train, x_test, y_train, y_test, feats, df_time

        elif submit == 'submission':
            df_time = {'time':X.index.tolist()}
            return X, feats, df_time

        elif submit == 'train_x_sub':
            df_time = {'time':X.index.tolist()}
            return X, Y, feats, df_time
