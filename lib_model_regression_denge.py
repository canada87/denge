import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
# sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
import matplotlib.pyplot as plt
from bokeh.plotting import figure

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import json

from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
import xgboost as xgb
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from lib_preproc_denge import preprocess_data

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.stats.stattools import durbin_watson

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

from fbprophet import Prophet


class learning_reg:
    def __init__ (self, SEED = 222):
        self.SEED = SEED

    def prophet_model_fit(self, train_in, test_in, df_time, city, verbose = 1):
        train = train_in.copy()
        test = test_in.copy()

        train['ds'] = df_time['train']
        train.rename(columns={'total_cases':'y'}, inplace=True)

        test['ds'] = df_time['test']
        test.rename(columns={'total_cases':'y'}, inplace=True)

        cols = []
        for col in train.columns:
            if col != 'y' and col != 'ds':
                cols.append(col)

        m = Prophet(changepoint_prior_scale = 0.01)
        for col in cols:
            m.add_regressor(col)
        m.fit(train)

        # future = pd.concat([train, test], ignore_index = True, axis=0)
        forecast = m.predict(test.drop(columns = 'y'))
        # st.write(forecast)
        # forecast['yhat'] = forecast['yhat']/forecast['trend']
        if city == 1:
            forecast['yhat'] = forecast['yhat']/forecast['trend'] + forecast['yearly'] + forecast['extra_regressors_additive'] + forecast['additive_terms']

        if verbose == 1:
            m.plot(forecast)
            test.index = test['ds']
            test['y'].plot(label = 'True', color = 'red')
            plt.legend(loc='upper left', fontsize=8)
            st.pyplot()
            m.plot_components(forecast)
            st.pyplot()

        predict_matrix = pd.DataFrame()
        predict_matrix['Prophet'] = forecast['yhat']
        return predict_matrix

    def varmax_model_fit(self, x_train, x_test, df_time, oreder = (1, 0), col_exog=[], verbose = 1):
        if col_exog:
            exo_train = pd.DataFrame()
            exo_test = pd.DataFrame()
            for col in col_exog:
                exo_train[col] = x_train[col]
                x_train.drop([col], axis=1, inplace = True)
                exo_test[col] = x_test[col]
                x_test.drop([col], axis=1, inplace = True)

            model = VARMAX(x_train, order=oreder, exog=exo_train)
        else:
            model = VARMAX(x_train, order=oreder)

        result = model.fit()
        out = durbin_watson(result.resid)
        df_results = pd.DataFrame()
        for col, val in zip(x_train.columns, out):
            df_results[col] = [round(val, 2)]
        if verbose == 1:
            st.subheader('durbin_watson test')
            st.write('the closer the result is to 2 then there is no correlation, the closer to 0 or 4 then correlation implies')
            st.write(df_results.T)

        if col_exog:
            df_forecast = result.forecast(steps=x_test.shape[0], exog = exo_test)
        else:
            df_forecast = result.forecast(steps=x_test.shape[0])

        df_forecast.index = df_time['test']
        df_forecast.columns = x_test.columns
        x_test.index = df_time['test']
        if verbose == 1:
            st.write(df_forecast)
            for i, col in enumerate(x_test):
                fig = ds().nuova_fig(555+i)
                st.subheader(col)
                df_forecast[col].plot(label = 'Predicition')
                x_test[col].plot(label = 'True')
                ds().legenda()
                st.pyplot(fig)
        return df_forecast

    def var_model_fit(self, x_train, x_test, df_time, lag, maxlag=None, verbose = 1):
        model = VAR(x_train)

        if maxlag != None:#studio hypersapce sul parametro lag
            vet_aic = []
            vet_bic = []
            vet_fpe = []
            vet_hqic = []
            for i in range(maxlag):
                result = model.fit(i)
                vet_aic.append(result.aic)
                vet_bic.append(result.bic)
                vet_fpe.append(result.fpe)
                vet_hqic.append(result.hqic)
            df_results = pd.DataFrame()
            df_results['AIC'] = vet_aic
            df_results['BIC'] = vet_bic
            df_results['FPE'] = vet_fpe
            df_results['HQIC'] = vet_hqic
            if verbose == 1:
                st.subheader('durbin_watson test')
                st.write('the closer the result is to 2 then there is no correlation, the closer to 0 or 4 then correlation implies')
                st.write(df_results)
        else:# fit diretto su un valore specifico di lag
            result = model.fit(lag)
            out = durbin_watson(result.resid)
            df_results = pd.DataFrame()
            for col, val in zip(x_train.columns, out):
                df_results[col] = [round(val, 2)]
            if verbose == 1:
                st.subheader('durbin_watson test')
                st.write('the closer the result is to 2 then there is no correlation, the closer to 0 or 4 then correlation implies')
                st.write(df_results.T)

        forecast_input = x_train.values[-lag:]
        fc = result.forecast(y=forecast_input, steps=x_test.shape[0])#y rappresenta i valori del training su cui verra prioritizzata la predizione per il fututo
        df_forecast = pd.DataFrame(fc, index=df_time['test'], columns=x_test.columns)
        x_test.index = df_time['test']
        if verbose == 1:
            st.write(df_forecast)
            for i, col in enumerate(x_test):
                ds().nuova_fig(555+i)
                st.subheader(col)
                df_forecast[col].plot(label = 'Predicition')
                x_test[col].plot(label = 'True')
                ds().legenda()
                st.pyplot()
        return df_forecast

    def var_predict_sbs(self, x_train, x_test, df_time, lag = 5, oreder = (1,0), col_exog = [], model = 'VAR', verbose = 1):
        total_predict = []
        total_true = []

        df_train = pd.DataFrame(x_train['total_cases'].copy())
        df_train.index = df_time['train']

        for i in range(x_test.shape[0]):
            x_new = pd.DataFrame(x_test.iloc[i]).T
            df_time_new ={'test': [df_time['test'][i]]}
            if model == 'VAR':
                predicted = self.var_model_fit(x_train, x_new, df_time_new, lag = lag, verbose = 0)
            elif model == 'VARMAX':
                x_train = x_train[-lag:]
                predicted = self.varmax_model_fit(x_train, x_new, df_time_new, oreder = oreder, col_exog=[], verbose = 0)

            total_predict.append(predicted['total_cases'].values[0])
            total_true.append(x_new['total_cases'].values[0])

            x_new.drop(['total_cases'], axis=1, inplace=True)
            x_new['total_cases'] = predicted['total_cases']
            x_train = pd.concat([x_train, x_new], ignore_index = True)

        if verbose == 1:
            st.subheader('VAR')
            df_results = pd.DataFrame()
            df_results['predicted'] = total_predict
            df_results['true'] = total_true
            df_results.index = df_time['test']
            # st.write(df_results)
            ds().nuova_fig(444)
            df_train['total_cases'].plot(label = 'train')
            df_results['predicted'].plot(label = 'Predicition')
            df_results['true'].plot(label = 'True')
            ds().legenda()
            st.pyplot()
        predict_matrix = pd.DataFrame()
        predict_matrix['VAR'] = df_results['predicted']
        predict_matrix = predict_matrix.reset_index(drop = True)
        return predict_matrix

    def autoarima_model_fit(self, x_train, y_train, df_time):

        x_train.index = df_time
        y_train.index = df_time

        model = pm.auto_arima(y_train,
                              exogenous=x_train,
                              start_p=1,
                              start_q=1,
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=10,
                              max_q=10,
                              m=52,
                              d=None,           # let model determine 'd'
                              seasonal=True,
                              start_P=0,
                              D=1,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
        # print(model.summary())
        # model.plot_diagnostics()
        # st.pyplot()

        # fitted, confint = model.predict(n_periods=x_test.shape[0], return_conf_int=True, exogenous=x_test)
        #
        # # make series for plotting purpose
        # fitted_series = pd.Series(fitted, index=x_test.index)
        # lower_series = pd.Series(confint[:, 0], index=x_test.index)
        # upper_series = pd.Series(confint[:, 1], index=x_test.index)
        #
        # # Plot
        # plt.plot(y_train, label='train')
        # plt.plot(y_test, label='actual')
        # plt.plot(fitted_series, color='darkgreen', label='forecast')
        # plt.fill_between(lower_series.index,
        #                  lower_series,
        #                  upper_series,
        #                  color='k', alpha=.15)
        # plt.legend(loc='upper left', fontsize=8)
        # st.pyplot()

        return model

    def sarimax_model_fit(self, x_train, y_train, df_time):#, y_test, x_test, df_test):

        # x_test.index = df_test
        # y_test.index = df_test

        x_train.index = df_time
        y_train.index = df_time

        model = SARIMAX(y_train, exog=x_train, order=(0, 1, 0), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=-1)
        print(model_fit.summary())

        # fc = model_fit.forecast(y_test.shape[0], exog = x_test)
        # fc.index = x_test.index
        # plt.plot(y_test, label='actual')
        # plt.plot(fc, label='forecast')
        # plt.legend(loc='upper left', fontsize=8)
        # st.pyplot()
        return model_fit



    def negative_binomial_model_eval(self, x_train, x_test, y_train, y_test):
        train = x_train.copy()
        test = x_test.copy()
        train['total_cases'] = y_train
        test['total_cases'] = y_test
        # Step 1: specify the form of the model
        model_formula = train.columns[0]
        for i in range(1, len(train.columns)-1):
            model_formula = model_formula+" + "+train.columns[i]
        model_formula = train.columns[-1] + ' ~ ' + model_formula

        grid = 10 ** np.arange(-8, -3, dtype=np.float64)

        best_alpha = []
        best_score = 1000

        # Step 2: Find the best hyper parameter, alpha
        for alpha in grid:
            model = smf.glm(formula=model_formula, data=train, family=sm.families.NegativeBinomial(alpha=alpha))

            results = model.fit()
            predictions = results.predict(test).astype(int)
            score = eval_measures.meanabs(predictions, test.total_cases)

            if score < best_score:
                best_alpha = alpha
                best_score = score

        # st.write('best alpha = ', best_alpha)
        # st.write('best score = ', best_score)
        return best_alpha

    def negative_binomial_fit(self, xdata, ydata, best_alpha):
        data = xdata.copy()
        data['total_cases'] = ydata

        model_formula = data.columns[0]
        for i in range(1, len(data.columns)-1):
            model_formula = model_formula+" + "+data.columns[i]
        model_formula = data.columns[-1] + ' ~ ' + model_formula

        # # Step 4: refit on entire dataset
        model = smf.glm(formula=model_formula, data=data, family=sm.families.NegativeBinomial(alpha=best_alpha))
        # model = smf.glm(formula=model_formula, data=data, family=sm.families.Poisson())
        fitted_model = model.fit()
        return fitted_model

    def deep_learning_model(self, input_dl, dropout_val, type = 'str', active = 'linear', lost = 'mse'):

        if type == 'str': #preprocess data "time"
            model = keras.Sequential()
            # act = keras.layers.LeakyReLU(alpha=0.01)
            act = 'relu'
            model.add(keras.layers.Dense(4, activation=act, input_dim = input_dl))
            # model.add(keras.layers.Dropout(dropout_val))
            # model.add(keras.layers.Dense(2, activation=act))
            # model.add(keras.layers.Dropout(dropout_val))
            model.add(keras.layers.Dense(1, activation = active))
            opt = keras.optimizers.SGD(lr=0.04,momentum=0.9)
            # opt = 'adam'
            model.compile(optimizer = opt, loss = lost, metrics=['mae'])

        if type == 'vanilla': #preprocess data "multi_var"
            model = keras.Sequential()
            n_steps, n_features = input_dl
            model.add(keras.layers.LSTM(8, activation='relu', input_shape=(n_steps, n_features)))
            # model.add(keras.layers.Dropout(dropout_val))
            model.add(keras.layers.Dense(1, activation = active))
            opt = keras.optimizers.SGD(lr=0.05,momentum=0.7)
            # opt = keras.optimizers.Adam(learning_rate=0.01)
            # opt = 'adam'
            model.compile(optimizer = opt, loss = lost, metrics=['mae'])

        if type == 'stacked': #preprocess data "multi_var"
            model = keras.Sequential()
            n_steps, n_features = input_dl
            model.add(keras.layers.LSTM(8, activation='relu', return_sequences = True, input_shape=(n_steps, n_features)))
            # model.add(keras.layers.Dropout(dropout_val))
            model.add(keras.layers.LSTM(4, activation='relu'))
            # model.add(keras.layers.Dropout(dropout_val))
            model.add(keras.layers.Dense(1, activation = active))
            opt = keras.optimizers.SGD(lr=0.05,momentum=0.7)
            model.compile(optimizer = opt, loss = lost, metrics=['mae'])

        if type == 'Bidirectional': #preprocess data "multi_var"
            model = keras.Sequential()
            n_steps, n_features = input_dl
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(16, activation='relu'), input_shape=(n_steps, n_features)))
            # model.add(keras.layers.Dropout(dropout_val))
            model.add(keras.layers.Dense(1, activation = active))
            opt = keras.optimizers.SGD(lr=0.005,momentum=0.9)
            model.compile(optimizer = opt, loss = lost, metrics=['mae'])

        if type == 'CNN_LSTM': #preprocess data "CNN"
            model = keras.Sequential()
            n_steps, n_features = input_dl

            model.add(keras.layers.TimeDistributed(keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
            model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=2)))
            model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))

            model.add(keras.layers.LSTM(4, activation='relu'))
            # model.add(keras.layers.Dropout(dropout_val))

            model.add(keras.layers.Dense(1, activation = active))
            opt = keras.optimizers.SGD(lr=0.02,momentum=0.5)
            model.compile(optimizer = opt, loss = lost, metrics=['mae'])

        if type == 'CNN_LSTM_stack': #preprocess data "CNN"
            model = keras.Sequential()
            n_steps, n_features = input_dl

            model.add(keras.layers.TimeDistributed(keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
            model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=2)))
            model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))

            model.add(keras.layers.LSTM(50, activation='relu', return_sequences = True))
            model.add(keras.layers.LSTM(10, activation='relu'))
            model.add(keras.layers.Dropout(dropout_val))

            model.add(keras.layers.Dense(1, activation = active))
            model.compile(optimizer = 'adam', loss = lost, metrics=['mae'])

        if type == 'convLSTM': #preprocess data "convLSTM"
            model = keras.Sequential()
            n_steps, n_features = input_dl

            model.add(keras.layers.ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(None, 1, n_steps, n_features)))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1, activation = active))
            opt = keras.optimizers.SGD(lr=0.005,momentum=0.9)
            model.compile(optimizer = opt, loss = lost, metrics=['mae'])

        return model

    def get_models(self, input_dl, type_dl, list_chosen, active, lost):
        """Generate a library of base learners."""

        # linreg=LinearRegression()
        # svrr = SVR(kernel='rbf')
        # dtr=DecisionTreeRegressor(random_state=self.SEED)
        # rf = RandomForestRegressor(n_estimators=100, bootstrap=True, random_state=self.SEED)
        # br = BaggingRegressor(n_estimators=300,random_state=self.SEED)
        # ada = AdaBoostRegressor(n_estimators=300,random_state=self.SEED)
        # gbr = GradientBoostingRegressor(n_estimators=300,random_state=self.SEED)
        # xgbr1 = xgb.XGBRegressor(n_estimators=100,random_state=self.SEED)

        linreg=LinearRegression(normalize=True, fit_intercept=True)
        dtr=DecisionTreeRegressor(random_state=222, min_samples_split=(0.018), min_samples_leaf= (0.007), max_depth=25)
        svrr = SVR(kernel='linear', epsilon=5)
        br = BaggingRegressor(n_estimators=350, max_samples=0.9, max_features=0.7, bootstrap=False, random_state=self.SEED)
        ada = AdaBoostRegressor(n_estimators=7, loss='exponential', learning_rate=0.01, random_state=self.SEED)
        rf = RandomForestRegressor(n_estimators=1000, max_depth= 30, max_leaf_nodes=1000, random_state=self.SEED)#, min_samples_split=0.002, max_features="auto", max_depth= 30, bootstrap=True, random_state=self.SEED)
        gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,random_state=self.SEED)
        xgbr1 = xgb.XGBRegressor(random_state=self.SEED)#n_estimators=100, max_depth = 4, random_state=self.SEED)
        mdl = LGBMRegressor(n_estimators=1000, learning_rate=0.01)

        las = Lasso()
        rid = Ridge()
        en = ElasticNet()
        huber = HuberRegressor(max_iter=2000)
        lasl = LassoLars(max_iter=2000, eps = 1, alpha=0.5, normalize=False)
        pa = PassiveAggressiveRegressor(C=1, max_iter=4000, random_state=self.SEED)
        sgd = SGDRegressor(max_iter=2000, tol=1e-3)

        knn = KNeighborsRegressor(n_neighbors=20)
        ex = ExtraTreeRegressor()
        exs = ExtraTreesRegressor(n_estimators=1000)

        dl = self.deep_learning_model(input_dl, dropout_val = 0.2, type = type_dl, active = active, lost = lost)

        models_temp = {

                    'deep learning': dl,

                    'BaggingRegressor': br,
                   'RandomForestRegressor': rf,
                   'GradientBoostingRegressor': gbr,
                    'XGBRegressor': xgbr1,
                   'LGBMRegressor':mdl,
                   'ExtraTreesRegressor': exs,


                  'LinearRegression': linreg,
                  'SVR': svrr,
                   'AdaBoostRegressor': ada,
                  'LassoLars': lasl,
                  'PassiveAggressiveRegressor': pa,
                  'SGDRegressor': sgd,

                   'DecisionTreeRegressor': dtr,

                  'lasso': las,
                  'ridge': rid,
                  'ElasticNet': en,
                  'HuberRegressor': huber,

                   'KNeighborsRegressor': knn,
                   'ExtraTreeRegressor': ex,
                  }

        models = dict()
        for model in list_chosen:
            if model in models_temp:
                models[model] = models_temp[model]
        st.write(models)

        return models


        # ████████ ██████   █████  ██ ███    ██ ██ ███    ██  ██████
        #    ██    ██   ██ ██   ██ ██ ████   ██ ██ ████   ██ ██
        #    ██    ██████  ███████ ██ ██ ██  ██ ██ ██ ██  ██ ██   ███
        #    ██    ██   ██ ██   ██ ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
        #    ██    ██   ██ ██   ██ ██ ██   ████ ██ ██   ████  ██████


    def train_horizontal_ensamble(self, model_list, xtrain, ytrain, epochs):
        name_model = 'deep learning'
        model = model_list[name_model]

        for epo in range(epochs):
            fitModel = model.fit(xtrain, ytrain, epochs = 1, batch_size = int(xtrain.shape[0]/3), verbose = 1, shuffle = False)
            if epo>=int(epochs - epochs*0.1):
                model.save('modelli/model_'+str(epo)+'.h5')

        model_list = self.load_horizontal_ensamble(model_list, epochs)
        return model_list

    def load_horizontal_ensamble(self, model_list, epochs):
        all_models = list()
        for epoc in range(int(epochs - epochs*0.1), epochs):
            sing_model = keras.models.load_model('modelli/model_'+str(epoc)+'.h5')
            all_models.append(sing_model)
        model_list['deep learning'] = all_models
        return model_list

    def train_predict(self, model_list, xtrain, ytrain, epochs, val_perc = 0.9):
        fitModel = 0
        for i, (name_model, model) in enumerate(model_list.items()):
            if name_model == 'deep learning':
                if val_perc == 1:
                    model.fit(xtrain, ytrain, epochs = epochs, batch_size = int(xtrain.shape[0]/3), verbose = 1, shuffle = False)
                else:
                    validation_size = int(xtrain.shape[0]*val_perc)
                    x_val = xtrain[validation_size:]
                    y_val = ytrain[validation_size:]
                    x_train_red = xtrain[:validation_size]
                    y_train_red = ytrain[:validation_size]
                    st.write(y_train_red)
                    fitModel = model.fit(x_train_red, y_train_red, epochs = epochs, batch_size = int(x_train_red.shape[0]/3), verbose = 1, validation_data= (x_val, y_val), shuffle = False)
            else:
                model.fit(xtrain, ytrain)
        return model_list, fitModel

    def hyper_training(self, model_list, x_data, y_data, dict_grid, filename = 'param_file'):
        dict_best_param = {}
        for i, (name_model, model) in enumerate(model_list.items()):
            model_random = GridSearchCV(estimator=model, param_grid=dict_grid[name_model], cv=3, verbose=2, n_jobs=-1)
            model_random.fit(x_data, y_data)
            dic_h = model_random.best_params_
            dict_best_param[name_model] = dic_h

        with open(filename, 'w') as f:
            json.dump(dict_best_param, f)

        return dict_best_param

    def rolling_training(self, model_list, X, Y, epochs, df_time, prep, frame_perc, validation = 'yes'):
        model = model_list['deep learning']
        df_df_time = pd.DataFrame()
        df_df_time['time'] = df_time['time']

        frame_size = int(X.shape[0]*frame_perc)
        last_step = round(X.shape[0]/frame_size)*frame_size - X.shape[0]

        score_data = []

        for step in range(round(X.shape[0]/frame_size)):
            start_frame = step*frame_size
            end_frame = start_frame + frame_size
            x_frame = X[start_frame:end_frame]
            y_frame = Y[start_frame:end_frame]
            df_df_time_frame = df_df_time[start_frame:end_frame]

            if validation == 'yes':
                y_pred = pd.DataFrame()
                y_true = pd.DataFrame()
                y_past = pd.DataFrame()

                x_train, x_val, y_train, y_val, time_train, time_val = train_test_split(x_frame, y_frame, df_df_time_frame, shuffle = False, test_size = 0.3)
                fitModel = model.fit(x_train, y_train, epochs = epochs, batch_size = int(x_train.shape[0]/3), verbose = 1, validation_data= (x_val, y_val), shuffle = False)

                y_pred['deep learning'] = model.predict(x_val)[:,0]
                y_pred.index = time_val['time']

                y_true['validation'] = y_val
                y_true.index = time_val['time']

                y_past['train'] = y_train
                y_past.index = time_train['time']

                self.plot_andamenti_dl(fitModel)
                ds().nuova_fig(567)
                y_pred['deep learning'].plot(label = 'prediction')
                y_true['validation'].plot(label = 'True')
                y_past['train'].plot(label = 'train')
                ds().legenda()
                st.pyplot()

                Predict_matrix = {'deep learning' : y_pred['deep learning'].ewm(span = 5).mean()}
                y_val_norm = prep.scale_backY(y_true['validation'])
                prep.scale_back_models(Predict_matrix)

                score = self.score_models(y_val_norm, Predict_matrix, verbose = 0)
                score_data.append(score['MAE'])
                st.dataframe(score)

            else:
                model.fit(x_frame, y_frame, epochs = epochs, batch_size = int(x_frame.shape[0]/3), verbose = 1, shuffle = False)

        if validation == 'yes':
            media =0
            for i in range(len(score_data)):
                media = score_data[i] + media
            media = media/len(score_data)
            st.write('MAE', media)

        if last_step != 0:
            x_frame = X[last_step:]
            y_frame = Y[last_step:]
            if validation != 'yes':
                model.fit(x_frame, y_frame, epochs = epochs, batch_size = int(x_frame.shape[0]/3), verbose = 1, shuffle = False)

        model_list['deep learning'] = model
        return model_list

        # ███████  ██████  ██████  ███████  ██████  █████  ███████ ████████
        # ██      ██    ██ ██   ██ ██      ██      ██   ██ ██         ██
        # █████   ██    ██ ██████  █████   ██      ███████ ███████    ██
        # ██      ██    ██ ██   ██ ██      ██      ██   ██      ██    ██
        # ██       ██████  ██   ██ ███████  ██████ ██   ██ ███████    ██


    def predict_matrix_generator(self, models, xtest, verbose = 0):
        def set_to_0(num):
            if num>10000:
                return 0
            else:
                return num

        if verbose == 1:
            st.subheader('predict')
        Predict_matrix = pd.DataFrame()
        cols = list()
        for i, (name_model, model) in enumerate(models.items()):

            if name_model == 'SARIMAX':
                Predict_matrix[name_model] = model.forecast(xtest.shape[0], exog = xtest).reset_index(drop = True)
            elif name_model == 'AUTOARIMA':
                Predict_matrix[name_model], confint = model.predict(n_periods=xtest.shape[0], return_conf_int=True, exogenous=xtest)
            elif name_model == 'deep learning':
                Predict_matrix[name_model] = model.predict(xtest)[:,0]
            elif name_model == 'NegativeBinomial':
                Predict_matrix[name_model] = model.predict(xtest).to_numpy()
            else:
                Predict_matrix[name_model] = model.predict(xtest)

            # Predict_matrix[name_model] = list(map(set_to_0, Predict_matrix[name_model]))
            cols.append(name_model)
        Predict_matrix['Ensamble'] = Predict_matrix.mean(axis=1)

        # vet_dic = dict()
        # for col in Predict_matrix.columns:
        #     vet_dic[col] = 'int'
        # Predict_matrix = Predict_matrix.astype(vet_dic)
        if verbose == 1:
            st.write(Predict_matrix)
        return Predict_matrix

    def predict_horizontal_ensamble(self, model_list, xtest):
        Predict_matrix = pd.DataFrame()
        for model in range(len(model_list['deep learning'])):
            Predict_matrix['deep learning '+str(model)] = model_list['deep learning'][model].predict(xtest)[:,0]
        Predict_matrix['Ensamble'] = Predict_matrix.mean(axis=1)
        return Predict_matrix

        # ███████  ██████  ██████  ██████  ███████
        # ██      ██      ██    ██ ██   ██ ██
        # ███████ ██      ██    ██ ██████  █████
        #      ██ ██      ██    ██ ██   ██ ██
        # ███████  ██████  ██████  ██   ██ ███████


    def score_models(self, y_test, Predict_matrix, verbose = 0):
        df_score = pd.DataFrame()
        for name_model in Predict_matrix:
            SS_residual = ((y_test - Predict_matrix[name_model])**2).sum()
            df_y = pd.DataFrame()
            df_y['y_test'] = y_test
            df_y['y_pred'] = Predict_matrix[name_model]
            df_y['diff'] = (y_test - Predict_matrix[name_model])

            SS_Total = ((y_test - np.mean(y_test))**2).sum()
            r_square = 1 - (float(SS_residual))/SS_Total

            mae = mean_absolute_error(y_test, Predict_matrix[name_model])
            mse = mean_squared_error(y_test, Predict_matrix[name_model])
            r2 = r2_score(y_test, Predict_matrix[name_model])
            df_score[name_model] = [round(mae,3), round(mse,3), round(r2,3), round(r_square,3), round(SS_residual,3)]

        df_score = df_score.T
        df_score.columns = ['MAE', 'MSE', 'R2', 'R2 a mano', 'residual']
        if verbose == 1:
            fig, ax1 = ds().nuova_fig(123)
            ds().titoli(titolo='', xtag='model', ytag='MAE')
            ds().dati(x = df_score.index.tolist(), y = df_score['MAE'].to_numpy(), scat_plot = 'scat', larghezza_riga = 15)
            modelli = df_score.index.tolist()
            plt.xticks(rotation=90)
            fig.tight_layout()
            st.pyplot()
        return df_score


        # ██████  ██       ██████  ████████
        # ██   ██ ██      ██    ██    ██
        # ██████  ██      ██    ██    ██
        # ██      ██      ██    ██    ██
        # ██      ███████  ██████     ██


    def plot_true_predict_realtion(self, y_pred, y_true, df_time, y_train, df_time_train, only_ensamble = 'no'):

        df_train = pd.DataFrame()
        df_train['y_train'] = y_train
        df_train['time'] = df_time_train
        df_train.set_index('time', inplace = True)

        y_pred['time'] = df_time
        y_pred.set_index('time', inplace = True)
        st.write(y_pred)

        df_true = pd.DataFrame()
        df_true['y_true'] = y_true
        df_true['time'] = df_time
        df_true.set_index('time', inplace = True)

        if only_ensamble == 'no':
            for i, model in enumerate(y_pred):
                st.subheader(model)
                ds().nuova_fig(150+i)
                ds().titoli(titolo='', xtag='true val', ytag='predict val')
                ds().dati(y_true, y_pred[model], scat_plot ='scat')
                plt.plot([0,y_true.max()],[0,y_true.max()], linestyle = '--')
                st.pyplot()

                ds().nuova_fig(90+i)
                df_train['y_train'].plot(label = 'train')
                df_true['y_true'].plot(label = 'True')
                y_pred[model].plot(label = 'Predicition')
                ds().legenda()
                st.pyplot()

                # sns.regplot(y_true, y_pred[model], color='teal', label = model, marker = 'x')
                # st.pyplot()
        else:
            st.subheader('Ensamble')
            # y_pred['Ensamble'] = y_pred['Ensamble'].shift(-2)
            # y_pred.fillna(method='ffill', inplace=True)
            # y_pred.fillna(method='bfill', inplace=True)

            ds().nuova_fig(150)
            ds().titoli(titolo='', xtag='true val', ytag='predict val')
            ds().dati(y_true, y_pred['Ensamble'], scat_plot ='scat')
            plt.plot([0,y_true.max()],[0,y_true.max()], linestyle = '--')
            st.pyplot()

            ds().nuova_fig(90)
            df_train['y_train'].plot(label = 'train')
            df_true['y_true'].plot(label = 'True')
            y_pred['Ensamble'].plot(label = 'Predicition')
            ds().legenda()
            st.pyplot()

    def plot_andamenti_dl(self, fitModel):
        if fitModel != 0:
            history_dict = fitModel.history
            history_dict.keys()

            acc = history_dict['mean_absolute_error']
            val_acc = history_dict['val_mean_absolute_error']
            loss = history_dict['loss']
            val_loss = history_dict['val_loss']

            epochs = range(1, len(acc) + 1)

            ds().nuova_fig(1)
            ds().titoli(titolo="Training loss", xtag='Epochs', ytag='Loss', griglia=0)
            ds().dati(epochs, loss, descrizione = 'Training loss', colore='red')
            ds().dati(epochs, val_loss, descrizione = 'Validation loss')
            ds().dati(epochs, loss, colore='red', scat_plot ='scat', larghezza_riga =10)
            ds().dati(epochs, val_loss, scat_plot ='scat', larghezza_riga =10)
            ds().range_plot(bottomY =np.array(val_loss).mean()-np.array(val_loss).std()*6, topY = np.array(val_loss).mean()+np.array(val_loss).std()*6)
            ds().legenda()
            st.pyplot()


            p = figure(title='Training and validation accuracy', x_axis_label='Epochs', y_axis_label='MAE', y_range = (np.array(val_acc).mean()-np.array(val_acc).std()*6, np.array(val_acc).mean()+np.array(val_acc).std()*6))
            p.scatter(epochs, acc, legend_label='Training acc', line_width=1, color = 'red')
            p.line(epochs, acc, line_width=1, color = 'red')
            p.scatter(epochs, val_acc, legend_label='Validation acc', line_width=1, color = 'blue')
            p.line(epochs, val_acc, line_width=1, color = 'blue')
            st.bokeh_chart(p, use_container_width=True)
