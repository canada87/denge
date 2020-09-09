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


# ██   ██  ██████  ██████  ██ ███████  ██████  ███    ██ ████████  █████  ██          ███████ ███    ██ ███████  █████  ███    ███ ██      ██████  ███████
# ██   ██ ██    ██ ██   ██ ██    ███  ██    ██ ████   ██    ██    ██   ██ ██          ██      ████   ██ ██      ██   ██ ████  ████ ██      ██   ██ ██
# ███████ ██    ██ ██████  ██   ███   ██    ██ ██ ██  ██    ██    ███████ ██          █████   ██ ██  ██ ███████ ███████ ██ ████ ██ ██      ██████  █████
# ██   ██ ██    ██ ██   ██ ██  ███    ██    ██ ██  ██ ██    ██    ██   ██ ██          ██      ██  ██ ██      ██ ██   ██ ██  ██  ██ ██      ██   ██ ██
# ██   ██  ██████  ██   ██ ██ ███████  ██████  ██   ████    ██    ██   ██ ███████     ███████ ██   ████ ███████ ██   ██ ██      ██ ███████ ██████  ███████

# sys.exit()

st.write('**vanilla**: normalize (**yes**), num_weeks (**3**), epochs (**300**), cicli_media (**1**), average orizzontale (**no**)')

st.sidebar.button('repeat')
rem_feat = 'yes'
additional_feat_rem = 'yes'#keep it yes
average = 5
normalizzare = st.sidebar.radio('Adding normalizatino', ['yes', 'no'])
num_weeks = int(st.sidebar.text_input('number of weeks', 5))#3
epochs = int(st.sidebar.text_input('Epochs', 30))#300
side_selection = st.sidebar.radio('Select the type of analysis', ['idle', 'train', 'submit'])
cicli_media = int(st.sidebar.text_input('Number of cicle to average', 1))#1
orizzontale = st.sidebar.radio('average orizzontale', ['no', 'yes', 'rolling'])#no
frame_perc = float(st.sidebar.text_input('Portion of data per frame to roll', 0.2))

type_of_nn = st.sidebar.selectbox("Type of network", ('vanilla', 'stacked', 'Bidirectional', 'CNN_LSTM', 'CNN_LSTM_stack', 'convLSTM' ))
modalita = {'vanilla': 'multi_var',
            'stacked': 'multi_var',
            'Bidirectional': 'multi_var',
            'CNN_LSTM': 'CNN',
            'CNN_LSTM_stack': 'CNN',
            'convLSTM': 'convLSTM'}

city = st.sidebar.radio('city', [0,1])

#LSTM
list_chosen = [
    'deep learning',
]

#############################################################################

if side_selection == 'train':
    prep = preprocess_data(normalize = normalizzare)
    if orizzontale == 'rolling':
        X, Y, df_time, feature, week = prep.training(filename1, filename2,
                                                          type = modalita[type_of_nn],
                                                          submit ='submit',
                                                          master = 'yes',
                                                          city = city, rem_feat = rem_feat,
                                                          additional_feat_rem = additional_feat_rem,#keep it yes
                                                          adding_feature = 'no',
                                                          num_weeks = num_weeks,
                                                          perc = 0.2955,
                                                          month ='no',
                                                          verbose = 0)

        models_base = learning_reg().get_models((week, feature), type_of_nn, list_chosen, 'linear', 'mae')
        models_base = learning_reg().rolling_training(models_base, X, Y, epochs, df_time, prep, frame_perc, validation = 'yes')

    else:
        x_train, x_test, y_train, y_test, df_time, feature, week = prep.training(filename1, filename2,
                                                                                        type = modalita[type_of_nn],
                                                                                      submit ='train',
                                                                                      master = 'yes',
                                                                                      city = city, rem_feat = rem_feat,
                                                                                      additional_feat_rem = additional_feat_rem,#keep it yes
                                                                                      adding_feature = 'no',
                                                                                      num_weeks = num_weeks,
                                                                                      # perc = 0.2955,
                                                                                      perc = 0.22,
                                                                                      month ='no',
                                                                                      verbose = 0)
        # sys.exit()
        models_base = dict()
        st.header('LSTM')
        st.write('shape:', x_train.shape)
        st.write('weeks:', week)
        st.write('features:', feature)

        # sys.exit()

        models_base = learning_reg().get_models((week, feature), type_of_nn, list_chosen, 'linear', 'mae')

        if orizzontale == 'yes':
            # orizzontale
            models_base = learning_reg().train_horizontal_ensamble(models_base, x_train, y_train, epochs = epochs)
            # models_base = learning_reg().load_horizontal_ensamble(models_base, epochs = epochs)
            Predict_matrix = learning_reg().predict_horizontal_ensamble(models_base, x_test)
        elif orizzontale == 'no':
            # predizione diretta
            models_base, fitModel = learning_reg().train_predict(models_base, x_train, y_train, epochs = epochs, val_perc = 1)
            Predict_matrix = learning_reg().predict_matrix_generator(models_base, x_test)
            learning_reg().plot_andamenti_dl(fitModel)

        Predict_matrix = Predict_matrix.ewm(span = average).mean()
        y_train = prep.scale_backY(y_train)
        y_test = prep.scale_backY(y_test)
        prep.scale_back_models(Predict_matrix)

        score_data = learning_reg().score_models(y_test, Predict_matrix, verbose = 1)
        st.dataframe(score_data)
        learning_reg().plot_true_predict_realtion(Predict_matrix, y_test, df_time['test'], y_train, df_time['train'], only_ensamble = 'yes')


# ███████ ██    ██ ██████  ███    ███ ██ ███████ ███████ ██  ██████  ███    ██
# ██      ██    ██ ██   ██ ████  ████ ██ ██      ██      ██ ██    ██ ████   ██
# ███████ ██    ██ ██████  ██ ████ ██ ██ ███████ ███████ ██ ██    ██ ██ ██  ██
#      ██ ██    ██ ██   ██ ██  ██  ██ ██      ██      ██ ██ ██    ██ ██  ██ ██
# ███████  ██████  ██████  ██      ██ ██ ███████ ███████ ██  ██████  ██   ████

# sys.exit()
if side_selection == 'submit':
    for cicli in range(cicli_media):
        st.title('ciclo '+str(cicli))
        prep2 = preprocess_data(normalize = normalizzare)
        dic_pred_sub = dict()
        cities = ['sj', 'iq']
        for i in range(2):
            st.title(cities[i])
            st.subheader('train')
            X, Y, df_time_train, feature, week = prep2.training(filename1, filename2,
                                                          type = modalita[type_of_nn],
                                                          submit ='submit',
                                                          master = 'yes',
                                                          city = i, rem_feat = rem_feat,
                                                          additional_feat_rem = additional_feat_rem,#keep it yes
                                                          adding_feature = 'no',
                                                          num_weeks = num_weeks,
                                                          perc = 0.2955,
                                                          month ='no',
                                                          verbose = 0)

            models_base = learning_reg().get_models((week, feature), type_of_nn, list_chosen, 'linear', 'mae')

            if orizzontale == 'yes':
                models_base = learning_reg().train_horizontal_ensamble(models_base, X, Y, epochs = epochs)
            elif orizzontale == 'no':
                models_base, fitModel = learning_reg().train_predict(models_base, X, Y, epochs = epochs, val_perc = 1)
            elif orizzontale == 'rolling':
                models_base = learning_reg().rolling_training(models_base, X, Y, epochs, df_time_train, prep2, frame_perc, validation = 'no')

            st.subheader('predict')
            x_sub, y_sub, df_time, feature, week = prep2.training(filename0, filename3,
                                               master = 'no',
                                              type = modalita[type_of_nn],
                                              submit ='submit',
                                              city = i, rem_feat = 'yes',
                                              additional_feat_rem = 'yes',#keep it yes
                                              adding_feature = 'no',
                                              num_weeks = num_weeks,
                                              month ='no',
                                              verbose = 0)

            if orizzontale == 'yes':
                dic_pred_sub[cities[i]] = learning_reg().predict_horizontal_ensamble(models_base, x_sub)
            elif orizzontale != 'yes':
                dic_pred_sub[cities[i]] = learning_reg().predict_matrix_generator(models_base, x_sub)

            dic_pred_sub[cities[i]] = dic_pred_sub[cities[i]].ewm(span = average).mean()
            Y = prep2.scale_backY(Y)
            prep2.scale_back_models(dic_pred_sub[cities[i]])

            st.subheader('result')
            y_fake = [0 for i in range(dic_pred_sub[cities[i]].shape[0])]
            # for col in dic_pred_sub[cities[i]].columns:
            #     dic_pred_sub[cities[i]][col] = dic_pred_sub[cities[i]][col] - dic_pred_sub[cities[i]][col].min()

            learning_reg().plot_true_predict_realtion(dic_pred_sub[cities[i]], np.array(y_fake), df_time['time'], Y, df_time_train['time'], only_ensamble = 'yes')

        df_sj = pd.DataFrame(dic_pred_sub['sj'])
        df_iq = pd.DataFrame(dic_pred_sub['iq'])
        df_pred = pd.concat([df_sj, df_iq], axis=0)
        df_pred = df_pred.reset_index(drop = True)

        df = pd.read_csv(filename3)
        df.drop(['total_cases'], inplace=True, axis=1)
        df['total_cases'] = df_pred['Ensamble']
        df['total_cases'].fillna(method='ffill', inplace=True)
        df['total_cases'] = df['total_cases'].astype(int)
        df.to_csv('sottomissioni/ML/submission_lstm'+str(cicli)+'.csv', sep=',', index = False)

    dic_load = dict()
    df_medio = pd.DataFrame()
    st.header('averaging')
    for cicli in range(cicli_media):
        dic_load[cicli] = pd.read_csv('sottomissioni/ML/submission_lstm'+str(cicli)+'.csv')
        df_medio = pd.concat([df_medio, dic_load[cicli]['total_cases']], axis = 1)
    df_medio['total_cases_average'] = df_medio.mean(axis = 1)

    df = pd.read_csv(filename3)
    df.drop(['total_cases'], inplace=True, axis=1)
    df['total_cases'] = df_medio['total_cases_average'].astype(int)
    st.write(df)
    df.to_csv('sottomissioni/submission_lstm.csv', sep=',', index = False)



    #  ██████  ██████  ███    ██ ███████ ██████   ██████  ███    ██ ████████  ██████
    # ██      ██    ██ ████   ██ ██      ██   ██ ██    ██ ████   ██    ██    ██    ██
    # ██      ██    ██ ██ ██  ██ █████   ██████  ██    ██ ██ ██  ██    ██    ██    ██
    # ██      ██    ██ ██  ██ ██ ██      ██   ██ ██    ██ ██  ██ ██    ██    ██    ██
    #  ██████  ██████  ██   ████ ██      ██   ██  ██████  ██   ████    ██     ██████

    name1 = 'sottomissioni/submission_var_old_dnn_average_on_20'
    name2 = 'sottomissioni/submission_sj_lstm_iq_var_old_dnn_average_on_20_'

    df1 = pd.read_csv(name1+'.csv')
    df2 = pd.read_csv(name2+'.csv')

    st.write(df1)

    df1_sj = df1[df1['city'] == 'sj']
    df1_iq = df1[df1['city'] == 'iq']

    df2_sj = df2[df2['city'] == 'sj']
    df2_iq = df2[df2['city'] == 'iq']

    ds().nuova_fig(77)
    st.subheader('SJ')
    df1_sj['total_cases'].plot(label = name1)
    df2_sj['total_cases'].plot(label = name2)
    ds().legenda()
    st.pyplot()

    st.subheader('IQ')
    ds().nuova_fig(78)
    df1_iq['total_cases'].plot(label = name1)
    df2_iq['total_cases'].plot(label = name2)
    ds().legenda()
    st.pyplot()

    ##########################

#
# name1 = 'sottomissioni/submission_var_old_dnn_average_on_20'
# name2 = 'sottomissioni/submission_lstm'
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
#
# df_tot_sj = df1_sj.copy()
# df_tot_sj['total_cases'] = df2_sj['total_cases']
#
# df_tot_iq = df1_iq.copy()
# df_tot_iq['total_cases'] = df1_iq['total_cases']
#
#
#
# # df_tot_sj = df1_sj.copy()
# # df_tot_sj['total_cases'] = (df1_sj['total_cases'] + df2_sj['total_cases'])/2
# #
# # df_tot_iq = df1_iq.copy()
# # df_tot_iq['total_cases'] = (df1_iq['total_cases'] + df2_iq['total_cases'])/2
#
#
#
#
# df_pred = pd.concat([df_tot_sj, df_tot_iq], axis=0)
# df_pred = df_pred.reset_index(drop = True)
# df_pred['total_cases'] = df_pred['total_cases'].astype(int)
# st.write(df_pred)
# df_pred.to_csv('sottomissioni/submission_sj_lstm_iq_var_old_dnn_average_on_20_.csv', sep=',', index = False)
