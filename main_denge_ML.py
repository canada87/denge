import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
# sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
from AJ_draw import disegna as ds
import matplotlib.pyplot as plt
import statsmodels.api as sm

import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb

from sklearn.preprocessing import label_binarize

from lib_preproc_denge import preprocess_data
from lib_model_regression_denge import learning_reg

import seaborn as sns

filename0 = 'dati/dengue_features_test.csv'
filename1 = 'dati/dengue_features_train.csv'
filename2 = 'dati/dengue_labels_train.csv'
filename3 = 'dati/submission_format.csv'

# ███    ███ ██
# ████  ████ ██
# ██ ████ ██ ██
# ██  ██  ██ ██
# ██      ██ ███████

# sys.exit()
st.write('**deep learning**: normalize (**yes**), adding_feature (**no**), epochs (**1000**), cicli_media (**20**), average orizzontale (**no**), mesi (**no**), anni (**no**), settimane (**no**)')
#########################################################################################################

average = 5



normalizzare = st.sidebar.radio('Adding normalizatino', ['yes', 'no'])#yes
adding_feature = st.sidebar.radio('Adding features', ['no', 'yes'])#no
epochs = int(st.sidebar.text_input('Epochs', 30))#1000
side_selection = st.sidebar.radio('Select the type of analysis', ['idle', 'train', 'submit'])
num_weeks = int(st.sidebar.text_input('number of weeks', 5))

rolling = st.sidebar.radio('rolling', ['no', 'yes'])#no
frame_perc = float(st.sidebar.text_input('Portion of data per frame to roll', 0.2))

list_models = [
                'deep learning',
                    'BaggingRegressor',
                   'RandomForestRegressor',
                   'GradientBoostingRegressor',
                    'XGBRegressor',
                   'ExtraTreesRegressor',
                   'LGBMRegressor',
                   'NegativeBinomial',
                   'SARIMAX',
                   'SARIMAX_auto',
]

list_chosen = st.sidebar.multiselect('models ', list_models)
mesi = st.sidebar.radio('Adding mesi', ['no', 'yes'])#no
anni = st.sidebar.radio('Adding anni', ['no', 'yes'])#no
settimane = st.sidebar.radio('Adding settimane', ['no', 'yes'])#no
cicli_media = int(st.sidebar.text_input('Number of cicle to average', 1))#20
city = st.sidebar.radio('city', [0,1])#no

if list_chosen:

    #########################################################################################################
    if side_selection == 'train':
        prep2 = preprocess_data(normalize = normalizzare)
        if rolling == 'yes':
            X, Y, df_time = prep2.training(filename1, filename2,
                                                                    submit ='train_x_sub',
                                                                    type = 'time',
                                                                    master = 'yes',
                                                                    city = city, rem_feat = 'yes',
                                                                    additional_feat_rem = 'yes',#keep it yes
                                                                    adding_feature = adding_feature,
                                                                    num_weeks = num_weeks,
                                                                    perc=0.11,
                                                                    month =mesi,
                                                                    year = anni,
                                                                    weeks = settimane,
                                                                    verbose = 0)
            models_base = dict()
            st.header('base regression')
            st.write('weeks:', num_weeks)
            st.write('shape:', X.shape)
            st.write('features:', X.shape[1])

            models_base = learning_reg().get_models(X.shape[1], 'str', list_chosen, 'linear', 'mse')
            models_base = learning_reg().rolling_training(models_base, X, Y, epochs, df_time, prep2, frame_perc, validation = 'yes')
        else:
            x_train, x_test, y_train, y_test, df_time = prep2.training(filename1, filename2,
                                                                                        submit ='train',
                                                                                        type = 'time',
                                                                                        master = 'yes',
                                                                                        city = city, rem_feat = 'yes',
                                                                                        additional_feat_rem = 'yes',#keep it yes
                                                                                        adding_feature = adding_feature,
                                                                                        num_weeks = num_weeks,
                                                                                        perc=0.11,
                                                                                        month =mesi,
                                                                                        year = anni,
                                                                                        weeks = settimane,
                                                                                        verbose = 0)

            models_base = dict()
            st.header('base regression')
            st.write('weeks:', num_weeks)
            st.write('shape:', x_train.shape)
            st.write('features:', x_train.shape[1])

            models_base = learning_reg().get_models(x_train.shape[1], 'str', list_chosen, 'linear', 'mse')
            models_base, fitModel = learning_reg().train_predict(models_base, x_train, y_train, epochs = epochs)
            learning_reg().plot_andamenti_dl(fitModel)

            #AUTOARIMA
            if 'SARIMAX_auto' in list_chosen:
                model_arima = learning_reg().autoarima_model_fit(x_train, y_train, df_time['train'])
                models_base['SARIMAX_auto'] = model_arima

            #SARIMAX
            if 'SARIMAX' in list_chosen:
                model_sarimax = learning_reg().sarimax_model_fit(x_train, y_train, df_time['train'])
                models_base['SARIMAX'] = model_sarimax

            #negative binomial
            if 'NegativeBinomial' in list_chosen:
                alpha = learning_reg().negative_binomial_model_eval(x_train, x_test, y_train, y_test)
                st.write('alpha', alpha)
                model_NB = learning_reg().negative_binomial_fit(x_train, y_train, alpha)
                models_base['NegativeBinomial'] = model_NB

            Predict_matrix_w_enasmble = learning_reg().predict_matrix_generator(models_base, x_test, verbose = 0)
            Predict_matrix_w_enasmble = Predict_matrix_w_enasmble.ewm(span = average).mean()
            prep2.scale_back_models(Predict_matrix_w_enasmble)
            y_train = prep2.scale_backY(y_train)
            y_test = prep2.scale_backY(y_test)
            score_data = learning_reg().score_models(y_test, Predict_matrix_w_enasmble)
            st.dataframe(score_data)
            learning_reg().plot_true_predict_realtion(Predict_matrix_w_enasmble, y_test, df_time['test'], y_train, df_time['train'])


    # ███████ ██    ██ ██████  ███    ███ ██ ███████ ███████ ██  ██████  ███    ██
    # ██      ██    ██ ██   ██ ████  ████ ██ ██      ██      ██ ██    ██ ████   ██
    # ███████ ██    ██ ██████  ██ ████ ██ ██ ███████ ███████ ██ ██    ██ ██ ██  ██
    #      ██ ██    ██ ██   ██ ██  ██  ██ ██      ██      ██ ██ ██    ██ ██  ██ ██
    # ███████  ██████  ██████  ██      ██ ██ ███████ ███████ ██  ██████  ██   ████

    # sys.exit()
    if side_selection == 'submit':
        for cicli in range(cicli_media):
            st.title('ciclo '+str(cicli))
            prep_sub = preprocess_data(normalize = normalizzare)
            dic_pred_sub = dict()
            dic_overfit = dict()
            cities = ['sj', 'iq']
            for i in range(2):
                st.title(cities[i])
                X, Y, df_time_train = prep_sub.training(filename1, filename2,
                                                        type = 'time',
                                                          submit ='train_x_sub',
                                                          master = 'yes',
                                                          city = i, rem_feat = 'yes',
                                                          additional_feat_rem = 'yes',#keep it yes
                                                          adding_feature = adding_feature,
                                                          num_weeks = num_weeks,
                                                          month =mesi,
                                                          year = anni,
                                                          weeks = settimane,
                                                          verbose =0)

                models_base = learning_reg().get_models(X.shape[1], 'str', list_chosen, 'linear', 'mse')

                if rolling == 'yes':
                    models_base = learning_reg().rolling_training(models_base, X, Y, epochs, df_time_train, prep_sub, frame_perc, validation = 'no')
                else:
                    models_base, fitmodel = learning_reg().train_predict(models_base, X, Y, epochs = epochs)

                #AUTOARIM
                if 'SARIMAX_auto' in list_chosen:
                    model_arima = learning_reg().autoarima_model_fit(X, Y, df_time['time'])
                    models_base['SARIMAX_auto'] = model_arima

                #SARIMAX
                if 'SARIMAX' in list_chosen:
                    model_sarimax = learning_reg().sarimax_model_fit(X, Y, df_time['time'])
                    models_base['SARIMAX'] = model_sarimax

                if 'NegativeBinomial' in list_chosen:
                    model_NB = learning_reg().negative_binomial_fit(X, Y, 0.0001)
                    models_base['NegativeBinomial'] = model_NB

                # st.subheader('overfit check')
                # dic_overfit[cities[i]] = learning_reg().predict_matrix_generator(models_base, X, verbose=1)
                # dic_overfit[cities[i]] = dic_overfit[cities[i]].ewm(span = average).mean()
                # prep_sub.scale_back_models(dic_overfit[cities[i]])
                # Y = prep_sub.scale_backY(Y)
                # learning_reg().plot_true_predict_realtion(dic_overfit[cities[i]], Y, df_time['time'])

                x_sub, df_time = prep_sub.training(filename0, filename3,
                                                        master = 'no',
                                                        type = 'time',
                                                          submit ='submission',
                                                          city = i, rem_feat = 'yes',
                                                          additional_feat_rem = 'yes',#keep it yes
                                                          adding_feature = adding_feature,
                                                          num_weeks = num_weeks,
                                                          month =mesi,
                                                          year = anni,
                                                          weeks = settimane,
                                                          verbose = 0)

                dic_pred_sub[cities[i]] = learning_reg().predict_matrix_generator(models_base, x_sub, verbose=1)
                dic_pred_sub[cities[i]] = dic_pred_sub[cities[i]].ewm(span = average).mean()
                Y = prep_sub.scale_backY(Y)
                prep_sub.scale_back_models(dic_pred_sub[cities[i]])

                st.subheader('result')
                y_fake = [0 for i in range(dic_pred_sub[cities[i]].shape[0])]
                # for col in dic_pred_sub[cities[i]].columns:
                #     dic_pred_sub[cities[i]][col] = dic_pred_sub[cities[i]][col] - dic_pred_sub[cities[i]][col].min()
                learning_reg().plot_true_predict_realtion(dic_pred_sub[cities[i]], np.array(y_fake), df_time['time'], Y, df_time_train['time'], only_ensamble = 'no')


            df_sj = pd.DataFrame(dic_pred_sub['sj'])
            df_iq = pd.DataFrame(dic_pred_sub['iq'])
            df_pred = pd.concat([df_sj, df_iq], axis=0)
            df_pred = df_pred.reset_index(drop = True)

            df = pd.read_csv(filename3)
            df.drop(['total_cases'], inplace=True, axis=1)
            df['total_cases'] = df_pred['Ensamble'].astype(int)
            df.to_csv('sottomissioni/ML/submission_dnn'+str(cicli)+'.csv', sep=',', index = False)

        dic_load = dict()
        df_medio = pd.DataFrame()
        st.header('averaging')
        for cicli in range(cicli_media):
            dic_load[cicli] = pd.read_csv('sottomissioni/ML/submission_dnn'+str(cicli)+'.csv')
            df_medio = pd.concat([df_medio, dic_load[cicli]['total_cases']], axis = 1)
        df_medio['total_cases_average'] = df_medio.mean(axis = 1)

        df = pd.read_csv(filename3)
        df.drop(['total_cases'], inplace=True, axis=1)
        df['total_cases'] = df_medio['total_cases_average'].astype(int)
        st.write(df)
        df.to_csv('sottomissioni/submission_ML.csv', sep=',', index = False)

    #  ██████  ██████  ███    ██ ███████ ██████   ██████  ███    ██ ████████  ██████
    # ██      ██    ██ ████   ██ ██      ██   ██ ██    ██ ████   ██    ██    ██    ██
    # ██      ██    ██ ██ ██  ██ █████   ██████  ██    ██ ██ ██  ██    ██    ██    ██
    # ██      ██    ██ ██  ██ ██ ██      ██   ██ ██    ██ ██  ██ ██    ██    ██    ██
    #  ██████  ██████  ██   ████ ██      ██   ██  ██████  ██   ████    ██     ██████

        name1 = 'sottomissioni/submission_var_old_dnn_average_on_20'
        name2 = 'sottomissioni/submission_ML'

        df1 = pd.read_csv(name1+'.csv')
        df2 = pd.read_csv(name2+'.csv')

        df1_sj = df1[df1['city'] == 'sj']
        df1_iq = df1[df1['city'] == 'iq']

        df2_sj = df2[df2['city'] == 'sj']
        df2_iq = df2[df2['city'] == 'iq']

        ds().nuova_fig(77)
        df1_sj['total_cases'].plot(label = name1)
        df2_sj['total_cases'].plot(label = name2)
        ds().legenda()
        st.pyplot()

        ds().nuova_fig(78)
        df1_iq['total_cases'].plot(label = name1)
        df2_iq['total_cases'].plot(label = name2)
        ds().legenda()
        st.pyplot()

        ##########################

# name1 = 'sottomissioni/submission_var_old_dnn_average_on_20'
# name2 = 'sottomissioni/submission_dnn'
#
# df1 = pd.read_csv(name1+'.csv')
# df2 = pd.read_csv(name2+'.csv')
#
# df1_sj = df1[df1['city'] == 'sj']
# df1_iq = df1[df1['city'] == 'iq']
#
# df2_sj = df2[df2['city'] == 'sj']
# df2_iq = df2[df2['city'] == 'iq']
#
# df_tot_sj = pd.DataFrame()
# df_tot_iq = pd.DataFrame()
#
# #
# # df_tot_sj = df1_sj.copy()
# # df_tot_sj['total_cases'] = (df1_sj['total_cases'] + df2_sj['total_cases'])/2
# #
# # df_tot_iq = df1_iq.copy()
# # df_tot_iq['total_cases'] = df1_iq['total_cases']
# #
#
#
# df_tot_sj = df1_sj.copy()
# df_tot_sj['total_cases'] = (df1_sj['total_cases'] + df2_sj['total_cases'])/2
#
# df_tot_iq = df1_iq.copy()
# df_tot_iq['total_cases'] = (df1_iq['total_cases'] + df2_iq['total_cases'])/2
#
#
#
#
# df_pred = pd.concat([df_tot_sj, df_tot_iq], axis=0)
# df_pred = df_pred.reset_index(drop = True)
# df_pred['total_cases'] = df_pred['total_cases'].astype(int)
# st.write(df_pred)
# df_pred.to_csv('sottomissioni/submission_var_old_dnn_average_on_20_ML.csv', sep=',', index = False)
