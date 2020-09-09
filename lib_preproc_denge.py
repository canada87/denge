import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import coint_johansen

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

    def __init__(self, normalize = 'yes', differenziato = 'no'):
        self.normalize = normalize
        self.scalerX = StandardScaler()
        self.scalerY = StandardScaler()
        self.differenziato = differenziato


    def type_of_file(self, file):
        dict_prova = {}
        dict_prova = {'type': file}
        st.write(dict_prova)


        # ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████
        #    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██
        #    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███
        #    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
        #    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████


    def training(self, filename, filename1, submit ='train', type = 'time', city = 0, rem_feat = 'no', additional_feat_rem = 'no',
                 adding_feature = 'no', num_weeks = 1, master = 'yes', perc=0.295,
                 weeks = 'no', month = 'no', year = 'no', verbose = 1):
        df = pd.read_csv(filename)
        df_labels = pd.read_csv(filename1)
        df_conc = pd.merge(df, df_labels, on=['city','year','weekofyear'])

        #seleziona la citta
        if city == 0:
            df_conc = df_conc[df_conc['city'] == 'sj']
        else:
            df_conc = df_conc[df_conc['city'] == 'iq']
            if master == 'yes':
                drop_row = [i for i in range(936,1013)]
                df_conc.drop(index = drop_row, inplace= True)
        df_conc.drop(['city'], axis = 1, inplace=True)

        # if master == 'yes':
        #     drop_row = [i for i in range(936,1013)]
        #     df_conc.drop(index = drop_row, inplace= True)

        #riempi i nan
        # df_conc.interpolate(method='linear', limit_direction='forward', axis = 0, inplace=True)
        df_conc.fillna(method='ffill', inplace=True)
        # df_conc.dropna(inplace = True)

        #trasforma le date in modo leggibile
        df_conc["week_start_date"] = pd.to_datetime(df_conc['week_start_date'])

        #metti come indice la data
        df_conc.set_index('week_start_date', inplace = True)

        #resample per mesi
        # df_conc = df_conc.resample('M').sum()

        #rimuovi le colonne che hai salvato
        df_conc.drop(['weekofyear', 'year'], axis = 1, inplace=True)

        #feature study
        df_conc = self.feature_study(df_conc, city)

        #smoothing funcion
        # df_conc = df_conc.ewm(span = 8).mean()

        #normalizza
        # df_conc[df_conc.columns] = self.scaler.fit_transform(df_conc)

        #rimuovi le feature inutili
        feature_to_remove = self.feature_remove(rem_feat, additional_feat_rem)

        #differenziazione
        if self.differenziato == 'yes':
            self.pre_diff = df_conc['total_cases'].copy()
            for i, col in enumerate(df_conc.columns):
                df_conc[col] = (df_conc[col].diff(num_weeks))
            df_conc.fillna(method='bfill', inplace=True)
            df_conc = df_conc.ewm(span = 5).mean()

        #split
        X=df_conc.drop(feature_to_remove, axis=1)
        Y=df_conc['total_cases']

        #salvo indice
        list_date = pd.DataFrame(X.index)
        list_date['ind'] = pd.to_datetime(list_date['week_start_date'])
        list_date.set_index('ind', inplace = True)

        if self.normalize == 'yes':
            #normalize X
            colonne = X.columns
            if master == 'yes':
                X=self.scalerX.fit_transform(X)
            else:
                X=self.scalerX.transform(X)
            X = pd.DataFrame(X)
            X.columns = colonne

            #normalize Y
            if master == 'yes':
                Y=self.scalerY.fit_transform(Y.to_numpy().reshape(len(Y),1))[:,0]
            else:
                Y=self.scalerY.transform(Y.to_numpy().reshape(len(Y),1))[:,0]
            Y = pd.Series(Y)
            Y.name = 'total_cases'
            X.index = list_date.index.tolist()
            Y.index = list_date.index.tolist()

##############################################################################
        #anasili della periodicita
        if verbose == 1:
            X2_tot = X.copy()
            X2_tot['total_cases'] = Y
            st.write(X2_tot)

            self.grangers_causation_matrix(X2_tot, maxlag = 12, verbose = 1)
            self.cointegration_test(X2_tot)
            self.adfuller_test(X2_tot, signif=0.05)
            self.autocorrelation_test(X2_tot, lag = 30)
##############################################################################

        #adding feature
        if adding_feature == 'yes':
            X_pre = X.copy()
            for j in range(1, num_weeks):
                for i in range(len(X_pre.columns)):
                    X[X_pre.columns[i]+'_'+str(j)+'_week_back'] = X[X_pre.columns[i]].shift(j)
                    # X[X_pre.columns[i]+'_'+str(j)+'_week_diff'] = X[X_pre.columns[i]+'_'+str(j)+'_week_back'].diff()
            X.fillna(method='bfill', inplace=True)

        #introdurre una lag
        # for col in X.columns:
        #     X[col] = X[col].shift(5)
        # X.fillna(method='ffill', inplace=True)
        # X.fillna(method='bfill', inplace=True)

        #add settimane mesi anni
        if year == 'yes':
            X['year'] = list_date['week_start_date'].dt.year.to_numpy()
        if month == 'yes':
            X['month'] = list_date['week_start_date'].dt.month.to_numpy()
        if weeks == 'yes':
            X['weeks'] = [i for i in range(X.shape[0])]

        return self.split_data(X, Y, perc, num_weeks, type, submit)

        # ███████ ██████  ██      ██ ████████
        # ██      ██   ██ ██      ██    ██
        # ███████ ██████  ██      ██    ██
        #      ██ ██      ██      ██    ██
        # ███████ ██      ███████ ██    ██

    def split_data(self, X, Y, perc, num_weeks, type, submit):
        #multivariant split

        def split_sequences(sequences, n_steps):
            ind = sequences.index.tolist()
            for i in range(n_steps-1):
                ind.pop(i)
            sequences = sequences.to_numpy()
            X, y = list(), list()
            for i in range(len(sequences)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the dataset
                if end_ix > len(sequences):
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y), ind

        if type == 'time':#arima
            if submit == 'train':
                x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle = False, test_size = perc)
                df_time = {'train':x_train.index.tolist(), 'test':x_test.index.tolist()}
                x_train = x_train.reset_index(drop = True)
                x_test = x_test.reset_index(drop = True)
                y_train = y_train.reset_index(drop = True)
                y_test = y_test.reset_index(drop = True)
                return x_train, x_test, y_train, y_test, df_time

            elif submit == 'train_x_sub':
                df_time = {'time':X.index.tolist()}
                return X, Y, df_time

            elif submit == 'submission':
                df_time = {'time':X.index.tolist()}
                return X, df_time

        elif type == 'multi_var':
            n_steps = num_weeks
            X['total_cases'] = Y.to_numpy()
            X, Y, ind = split_sequences(X, n_steps)
            feature = X.shape[2]
            week = X.shape[1]

            if submit == 'train':
                x_train, x_test, y_train, y_test, ind_train, ind_test = train_test_split(X, Y, ind, shuffle = False, test_size = perc)
                y_train = pd.Series(y_train)
                y_train.name = 'total_cases'
                y_test = pd.Series(y_test)
                y_test.name = 'total_cases'
                df_time = {'train': ind_train, 'test': ind_test}
                return x_train, x_test, y_train, y_test, df_time, feature, week

            elif submit == 'submit':
                Y = pd.Series(Y)
                Y.name = 'total_cases'
                df_time = {'time':ind}
                return X, Y, df_time, feature, week

        elif type == 'CNN':
            n_steps = num_weeks

            X['total_cases'] = Y.to_numpy()
            # X.drop(X.head(1).index,inplace=True)
            # X.drop(X.tail(1).index,inplace=True)
            X, Y, ind = split_sequences(X, n_steps)
            feature = X.shape[2]
            n_seq = 2

            if submit == 'train':
                x_train, x_test, y_train, y_test, ind_train, ind_test = train_test_split(X, Y, ind, shuffle = False, test_size = perc)
                x_train = x_train.reshape((x_train.shape[0], n_seq, int(n_steps/2), feature))
                x_test = x_test.reshape((x_test.shape[0], n_seq, int(n_steps/2), feature))
                week = x_train.shape[2]
                y_train = pd.Series(y_train)
                y_train.name = 'total_cases'
                y_test = pd.Series(y_test)
                y_test.name = 'total_cases'
                df_time = {'train': ind_train, 'test': ind_test}
                return x_train, x_test, y_train, y_test, df_time, feature, week

            elif submit == 'submit':
                Y = pd.Series(Y)
                Y.name = 'total_cases'
                X = X.reshape((X.shape[0], n_seq, int(n_steps/2), feature))
                week = X.shape[2]
                df_time = {'time':ind}
                return X, Y, df_time, feature, week

        elif type == 'convLSTM':
            n_steps = num_weeks
            X['total_cases'] = Y.to_numpy()
            X.drop(X.head(1).index,inplace=True)
            X, Y, ind = split_sequences(X, n_steps)
            feature = X.shape[2]
            n_seq = 2

            if submit == 'train':
                x_train, x_test, y_train, y_test, ind_train, ind_test = train_test_split(X, Y, ind, shuffle = False, test_size = perc)
                x_train = x_train.reshape((x_train.shape[0], n_seq, 1, int(n_steps/2), feature))
                x_test = x_test.reshape((x_test.shape[0], n_seq, 1, int(n_steps/2), feature))
                week = x_train.shape[3]
                y_train = pd.Series(y_train)
                y_train.name = 'total_cases'
                y_test = pd.Series(y_test)
                y_test.name = 'total_cases'
                df_time = {'train': ind_train, 'test': ind_test}
                return x_train, x_test, y_train, y_test, df_time, feature, week

            elif submit == 'submit':
                X = X.reshape((X.shape[0], n_seq, 1, int(n_steps/2), feature))
                week = X.shape[3]
                Y = pd.Series(Y)
                Y.name = 'total_cases'
                df_time = {'time':ind}
                return X, Y, df_time, feature, week

        elif type == 'VAR':
            X['total_cases'] = Y

            if submit == 'train':
                if self.differenziato == 'yes':
                    x_train, x_test, self.pre_diff_train, self.pre_diff_test = train_test_split(X, self.pre_diff, shuffle = False, test_size = perc)
                else:
                    x_train, x_test = train_test_split(X, shuffle = False, test_size = perc)

                df_time = {'train':x_train.index.tolist(), 'test':x_test.index.tolist() }
                x_train = x_train.reset_index(drop = True)
                x_test = x_test.reset_index(drop = True)
                return x_train, x_test, df_time

            elif submit == 'submit':
                df_time = {'time':X.index.tolist()}
                X = X.reset_index(drop = True)
                return X, df_time


# ███████ ███████  █████  ████████ ██████  ██    ██ ██████  ███████     ██████  ███████ ███    ███  ██████  ██    ██ ███████
# ██      ██      ██   ██    ██    ██   ██ ██    ██ ██   ██ ██          ██   ██ ██      ████  ████ ██    ██ ██    ██ ██
# █████   █████   ███████    ██    ██████  ██    ██ ██████  █████       ██████  █████   ██ ████ ██ ██    ██ ██    ██ █████
# ██      ██      ██   ██    ██    ██   ██ ██    ██ ██   ██ ██          ██   ██ ██      ██  ██  ██ ██    ██  ██  ██  ██
# ██      ███████ ██   ██    ██    ██   ██  ██████  ██   ██ ███████     ██   ██ ███████ ██      ██  ██████    ████   ███████


    def feature_remove(self, rem_feat = 'yes', additional_feat_rem = 'yes'):
        temperatura = [
                             'station_min_temp_c',
                              'reanalysis_min_air_temp_k',

                              'reanalysis_max_air_temp_k',
                            'station_max_temp_c',

                            'station_avg_temp_c',#main

                             'reanalysis_avg_temp_k',
                             'reanalysis_air_temp_k',

                           'station_diur_temp_rng_c',
                             'reanalysis_tdtr_k',
                            ]
        vegetazione = [
                              'ndvi_ne',
                              'ndvi_nw',
                              'ndvi_sw',
                             'ndvi_se',

                              'total_cases',
                             ]

        precipitazioni = [
                            'station_precip_mm',
                            'precipitation_amt_mm',
                             'reanalysis_sat_precip_amt_mm',
                            'reanalysis_precip_amt_kg_per_m2',
                            ]

        umidita = [
                            'reanalysis_dew_point_temp_k',
                            'reanalysis_specific_humidity_g_per_kg',
                             'reanalysis_relative_humidity_percent',
                              ]

        feature_remove_tot = np.concatenate((temperatura, vegetazione), axis=None)
        feature_remove_tot = np.concatenate((feature_remove_tot, precipitazioni), axis=None)
        feature_remove_tot = np.concatenate((feature_remove_tot, vegetazione), axis=None)
        feature_remove_tot = np.concatenate((feature_remove_tot, umidita), axis=None)

        if rem_feat == 'yes':
            return feature_remove_tot
        else:
            return ['total_cases']


# ███████  ██████  █████  ██      ███████     ██████   █████  ████████  █████
# ██      ██      ██   ██ ██      ██          ██   ██ ██   ██    ██    ██   ██
# ███████ ██      ███████ ██      █████       ██   ██ ███████    ██    ███████
#      ██ ██      ██   ██ ██      ██          ██   ██ ██   ██    ██    ██   ██
# ███████  ██████ ██   ██ ███████ ███████     ██████  ██   ██    ██    ██   ██

    def scale_backY(self, Y):
        if self.normalize == 'yes':
            Y=self.scalerY.inverse_transform(Y.to_numpy().reshape(len(Y),1))[:,0]
            Y = pd.Series(Y)
            Y.name = 'total_cases'
        return Y

    def scale_back_models(self, y_prob):
        if self.normalize == 'yes':
            for model in y_prob:
                y_prob[model] = self.scalerY.inverse_transform(y_prob[model].to_numpy().reshape(len(y_prob[model]),1))[:,0]



# ███████ ███████  █████  ████████ ██    ██ ██████  ███████     ███████ ████████ ██████  ██    ██ ██████  ██    ██
# ██      ██      ██   ██    ██    ██    ██ ██   ██ ██          ██         ██    ██   ██ ██    ██ ██   ██  ██  ██
# █████   █████   ███████    ██    ██    ██ ██████  █████       ███████    ██    ██████  ██    ██ ██   ██   ████
# ██      ██      ██   ██    ██    ██    ██ ██   ██ ██               ██    ██    ██   ██ ██    ██ ██   ██    ██
# ██      ███████ ██   ██    ██     ██████  ██   ██ ███████     ███████    ██    ██   ██  ██████  ██████     ██


    def tsplot(self, y, title, lags=None):
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0))
        hist_ax = plt.subplot2grid(layout, (0, 1))
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title(title, fontsize=12, fontweight='bold')
        y.plot(ax=hist_ax, kind='hist', bins=25)
        hist_ax.set_title('Histogram')
        sm.tsa.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        sm.tsa.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        sns.despine()
        plt.tight_layout()
        st.pyplot()

    def feature_study(self, data, city):

        # 1994-09-10
        # 1994-12-24
        # st.write(data['ndvi_ne']['1994-09-10'])
        data = data.replace({'ndvi_nw': {0.076225: -0.1}})
        data = data.replace({'ndvi_ne': {0.04585: -0.1}})
        if city == 1:
            soglia = 0.195
        else:
            soglia = -0.05
        df_vegetazione = pd.DataFrame()
        df_vegetazione['ndvi_ne_water'] = (data['ndvi_ne'].ewm(span = 4).mean() < soglia).astype(int)
        df_vegetazione['ndvi_nw_water'] = (data['ndvi_nw'].ewm(span = 4).mean() < soglia).astype(int)
        df_vegetazione['ndvi_sw_water'] = (data['ndvi_sw'].ewm(span = 4).mean() < soglia).astype(int)
        df_vegetazione['ndvi_se_water'] = (data['ndvi_se'].ewm(span = 4).mean() < soglia).astype(int)
        # data['vegetazione'] = df_vegetazione['ndvi_nw_water']+df_vegetazione['ndvi_ne_water']
        # data['vegetazione'] = ((data['ndvi_ne'] + data['ndvi_nw'])/2).ewm(span = 4).mean()

        data['vegetazione'] = ((data['ndvi_ne'] + data['ndvi_nw'] + data['ndvi_sw'] + data['ndvi_se'])/4).ewm(span = 4).mean()

        # ds().nuova_fig(1)
        # df_vegetazione['ndvi_ne_water'].plot(label = 'ndvi_ne_water')
        # data['ndvi_ne'].ewm(span = 4).mean().plot(label = 'ndvi_ne')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(2)
        # df_vegetazione['ndvi_nw_water'].plot(label = 'ndvi_nw_water')
        # data['ndvi_nw'].ewm(span = 4).mean().plot(label = 'ndvi_nw')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(2)
        # df_vegetazione['ndvi_se_water'].plot(label = 'ndvi_se_water')
        # data['ndvi_se'].ewm(span = 4).mean().plot(label = 'ndvi_se')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(2)
        # df_vegetazione['ndvi_sw_water'].plot(label = 'ndvi_sw_water')
        # data['ndvi_sw'].ewm(span = 4).mean().plot(label = 'ndvi_sw')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(5)
        # (data['vegetazione']).plot(label = 'ndvi_tot_water')
        # (data['total_cases']/data['total_cases'].max()).ewm(span = 4).mean().plot(label = 'ndvi_ne')
        # ds().legenda()
        # st.pyplot()


        # ds().nuova_fig(7)
        # data['total_cases'].ewm(span = 4).mean().plot(label = 'ndvi_ne')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(6)
        # # data['ndvi_ne'].ewm(span = 4).mean().plot(label = 'ndvi_ne')
        # # data['ndvi_nw'].ewm(span = 4).mean().plot(label = 'ndvi_nw')
        # # data['ndvi_sw'].ewm(span = 4).mean().plot(label = 'ndvi_sw')
        # # data['ndvi_se'].ewm(span = 4).mean().plot(label = 'ndvi_se')
        # data['ndvi_tot'].ewm(span = 4).mean().plot(label = 'ndvi_tot')
        # # data['ndvi_ne'].plot(label = 'ndvi_ne')
        # # data['ndvi_nw'].plot(label = 'ndvi_nw')
        # # data['ndvi_sw'].plot(label = 'ndvi_sw')
        # # data['ndvi_se'].plot(label = 'ndvi_se')
        # ds().legenda()
        # st.pyplot()

        # (data['reanalysis_tdtr_k']).plot(label = 'reanalysis_tdtr_k')
        # (data['reanalysis_avg_temp_k']-delta).plot(label = 'reanalysis_avg_temp_k')
        # ds().legenda()
        # st.pyplot()

        # for i, col in enumerate(data.columns):
        #     self.tsplot(data[col], col)
            # ds().nuova_fig(i)
            # data[col].plot(label = col)
            # ds().legenda()
            # st.pyplot()

        # data['range_temperatura'] = ((data['station_diur_temp_rng_c'] + data['reanalysis_tdtr_k'])/2).ewm(span = 4).mean()
        delta = data['reanalysis_air_temp_k'].mean() - data['station_avg_temp_c'].mean()
        data['temperature'] = ((data['station_avg_temp_c'] + (data['reanalysis_air_temp_k']-delta) + (data['reanalysis_avg_temp_k']-delta))/3).ewm(span = 4).mean()

        data['umidita'] = data['reanalysis_specific_humidity_g_per_kg'].ewm(span = 8).mean()

        data['precipitazioni'] = ((data['station_precip_mm']+data['precipitation_amt_mm']+data['reanalysis_sat_precip_amt_mm']+data['reanalysis_precip_amt_kg_per_m2'])/4).ewm(span = 4).mean()

        # ds().nuova_fig(7)
        # data['temperature'].ewm(span = 4).mean().plot(label = 'temperature')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(78)
        # data['umidita'].ewm(span = 4).mean().plot(label = 'umidita')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(79)
        # data['precipitazioni'].ewm(span = 4).mean().plot(label = 'precipitazioni')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(76)
        # data['vegetazione'].ewm(span = 4).mean().plot(label = 'vegetazione')
        # ds().legenda()
        # st.pyplot()
        #
        # ds().nuova_fig(75)
        # data['total_cases'].ewm(span = 4).mean().plot(label = 'total_cases')
        # ds().legenda()
        # st.pyplot()

        return data


        #  ██████ ██   ██ ███████  ██████ ██   ██      ██████  ██████  ███    ██      ██████  ██████  ██████  ██████  ███████ ██       █████  ████████ ██  ██████  ███    ██
        # ██      ██   ██ ██      ██      ██  ██      ██      ██    ██ ████   ██     ██      ██    ██ ██   ██ ██   ██ ██      ██      ██   ██    ██    ██ ██    ██ ████   ██
        # ██      ███████ █████   ██      █████       ██      ██    ██ ██ ██  ██     ██      ██    ██ ██████  ██████  █████   ██      ███████    ██    ██ ██    ██ ██ ██  ██
        # ██      ██   ██ ██      ██      ██  ██      ██      ██    ██ ██  ██ ██     ██      ██    ██ ██   ██ ██   ██ ██      ██      ██   ██    ██    ██ ██    ██ ██  ██ ██
        #  ██████ ██   ██ ███████  ██████ ██   ██      ██████  ██████  ██   ████      ██████  ██████  ██   ██ ██   ██ ███████ ███████ ██   ██    ██    ██  ██████  ██   ████

    def grangers_causation_matrix(self, data, maxlag = 12, test='ssr_chi2test', verbose=False):
        """Check Granger Causality of all possible combinations of the Time series.
        The rows are the response variable, columns are predictors. The values in the table
        are the P-Values. P-Values lesser than the significance level (0.05), implies
        the Null Hypothesis that the coefficients of the corresponding past values is
        zero, that is, the X does not cause Y can be rejected.

        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        """
        variables = data.columns
        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + '_x' for var in variables]
        df.index = [var + '_y' for var in variables]


        if verbose == 1:
            st.subheader('grangers causation matrix')
            st.write('if value < 0.05 then X causes Y')
            st.write(df)
            sns.heatmap(df, annot=True, annot_kws={'size': 8}, cmap=plt.cm.Blues, vmax=0.05,vmin=0, square=True, linewidths=0.1, linecolor="black")
            plt.xticks(rotation = 45)
            plt.tight_layout()
            st.pyplot()

        return df

    def cointegration_test(self, df, signif=0.05):
        """Perform Johanson's Cointegration Test and Report Summary"""
        st.subheader('cointegration test')
        out = coint_johansen(df,-1,5)
        d = {'0.90':0, '0.95':1, '0.99':2}
        traces = out.lr1
        cvts = out.cvt[:, d[str(1-signif)]]
        def adjust(val, length= 6): return str(val).ljust(length)

        # Summary
        # print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
        vet_name = []
        vet_test = []
        vet_c = []
        vet_sign = []
        for col, trace, cvt in zip(df.columns, traces, cvts):
            vet_name.append(adjust(col))
            vet_test.append(adjust(round(trace,2), 9))
            vet_c.append(adjust(cvt, 8))
            vet_sign.append(trace > cvt)
            # print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        df_cointegration = pd.DataFrame()
        df_cointegration['name'] = vet_name
        df_cointegration['test'] = vet_test
        df_cointegration['c(95%)'] = vet_c
        df_cointegration['signif'] = vet_sign
        st.dataframe(df_cointegration)

    def adfuller_test(self, df, signif=0.05):
        """Perform ADFuller to test for Stationarity of given series and print report"""
        st.subheader('ADF test')
        vet_signif = []
        vet_stat = []
        vet_lag = []
        vet_1 = []
        vet_5 = []
        vet_10 = []
        vet_p = []
        vet_true = []

        for col in df:
            series = df[col]
            r = adfuller(series, autolag='AIC')
            output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
            p_value = output['pvalue']

            vet_signif.append(signif)
            vet_stat.append(output["test_statistic"])
            vet_lag.append(output["n_lags"])
            vet_1.append(round(r[4]['1%'],3))
            vet_5.append(round(r[4]['5%'],3))
            vet_10.append(round(r[4]['10%'],3))
            vet_p.append(p_value)
            if p_value <= signif:
                vet_true.append('Stationary')
            else:
                vet_true.append('Non-Stationary')

        df_resuts = pd.DataFrame()
        df_resuts['significance level'] = vet_signif
        df_resuts['test statistic'] = vet_stat
        df_resuts['no. lag'] = vet_lag
        df_resuts['critival 1%'] = vet_1
        df_resuts['critival 5%'] = vet_5
        df_resuts['critival 10%'] = vet_10
        df_resuts['P-value'] = vet_p
        df_resuts['stat or not'] = vet_true
        df_resuts = df_resuts.T
        df_resuts.columns = df.columns
        st.write(df_resuts)

    def autocorrelation_test(self, data, lag):
        for i, col in enumerate(data.columns):
            self.tsplot(data[col], col, lags=30)


# ██ ███    ██ ██    ██ ███████ ██████  ████████     ██████  ██ ███████ ███████ ███████ ██████  ███████ ███    ██ ███████ ██  █████  ████████ ██  ██████  ███    ██
# ██ ████   ██ ██    ██ ██      ██   ██    ██        ██   ██ ██ ██      ██      ██      ██   ██ ██      ████   ██    ███  ██ ██   ██    ██    ██ ██    ██ ████   ██
# ██ ██ ██  ██ ██    ██ █████   ██████     ██        ██   ██ ██ █████   █████   █████   ██████  █████   ██ ██  ██   ███   ██ ███████    ██    ██ ██    ██ ██ ██  ██
# ██ ██  ██ ██  ██  ██  ██      ██   ██    ██        ██   ██ ██ ██      ██      ██      ██   ██ ██      ██  ██ ██  ███    ██ ██   ██    ██    ██ ██    ██ ██  ██ ██
# ██ ██   ████   ████   ███████ ██   ██    ██        ██████  ██ ██      ██      ███████ ██   ██ ███████ ██   ████ ███████ ██ ██   ██    ██    ██  ██████  ██   ████


    def invert_diff(self, train, predict):
        df_fc = predict.copy()
        df_fc = train.iloc[-1] + df_fc.cumsum()
        return df_fc
