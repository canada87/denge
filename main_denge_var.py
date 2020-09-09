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


# ██    ██  █████  ██████
# ██    ██ ██   ██ ██   ██
# ██    ██ ███████ ██████
#  ██  ██  ██   ██ ██   ██
#   ████   ██   ██ ██   ██


# sys.exit()
models_base = dict()
average = 5
week = 1
normalizzare = st.sidebar.radio('Adding normalizatino', ['no', 'yes'])
lag = int(st.sidebar.text_input('number of lag', 4))
side_selection = st.sidebar.radio('Select the type of analysis', ['idle', 'train', 'submit'])

mesi = st.sidebar.radio('Adding mesi', ['yes', 'no'])
anni = st.sidebar.radio('Adding anni', ['no', 'yes'])
settimane = st.sidebar.radio('Adding settimane', ['no', 'yes'])

if side_selection == 'train':
    prep = preprocess_data(normalize = normalizzare, differenziato = 'no')
    train, test, df_time = prep.training(filename1, filename2,
                                      type ='VAR',
                                      submit = 'train',
                                      master = 'yes',
                                      city = 0, rem_feat = 'yes',
                                      additional_feat_rem = 'yes',#keep it yes
                                      adding_feature = 'no',
                                      num_weeks = week,
                                      perc = 0.3,
                                      month =mesi,
                                      year = anni,
                                      weeks = settimane)
    # sys.exit()
    #step by step
    # predict_matrix = learning_reg().var_predict_sbs(train, test, df_time, lag = test.shape[0], oreder = (1,0), col_exog = [], model = 'VARMAX', verbose = 1)
    predict_matrix = learning_reg().var_predict_sbs(train, test, df_time, lag = lag, model = 'VAR', verbose = 1)

    df_score = learning_reg().score_models(test['total_cases'], predict_matrix)
    st.write(df_score)

    #single forecast
    # x_train, x_test, df_time = prep.training(filename1, filename2,
                                      # type ='VAR',
                                      # submit = 'train',
    #                                   master = 'yes',
    #                                   city = 0, rem_feat = rem_feat,
    #                                   additional_feat_rem = additional_feat_rem,#keep it yes
    #                                   adding_feature = 'no',
    #                                   num_weeks = week,
    #                                   perc = 0.1)
    #
    #
    # learning_reg().varmax_model_fit(x_train, x_test, df_time, col_exog=['precipitazioni'], verbose = 1)
    # learning_reg().var_model_fit(x_train, x_test, df_time, lag = x_test.shape[0], verbose = 1)


# ███████ ██    ██ ██████  ███    ███ ██ ████████     ██    ██  █████  ██████
# ██      ██    ██ ██   ██ ████  ████ ██    ██        ██    ██ ██   ██ ██   ██
# ███████ ██    ██ ██████  ██ ████ ██ ██    ██        ██    ██ ███████ ██████
#      ██ ██    ██ ██   ██ ██  ██  ██ ██    ██         ██  ██  ██   ██ ██   ██
# ███████  ██████  ██████  ██      ██ ██    ██          ████   ██   ██ ██   ██



if side_selection == 'submit':
    prep_sub = preprocess_data(normalize = normalizzare)
    dic_pred_sub = dict()

    cities = ['sj', 'iq']
    for i in range(2):
        st.title(cities[i])

        train, df_time_train = prep_sub.training(filename1, filename2,
                                      type ='VAR',
                                      submit = 'submit',
                                          master = 'yes',
                                          city = i, rem_feat = 'yes',
                                          additional_feat_rem = 'yes',#keep it yes
                                          adding_feature = 'no',
                                          num_weeks = week,
                                          perc = 0.3,
                                          month =mesi,
                                          year = anni,
                                          weeks = settimane,
                                          verbose =0)

        x_sub, df_time_sub = prep_sub.training(filename0, filename3,
                                           master = 'no',
                                            type ='VAR',
                                            submit = 'submit',
                                              city = i, rem_feat = 'yes',
                                              additional_feat_rem = 'yes',#keep it yes
                                              adding_feature = 'no',
                                              num_weeks = week,
                                              month =mesi,
                                              year = anni,
                                              weeks = settimane,
                                                  verbose = 0)
        x_sub['total_cases'] = 0

        df_time = {'test': df_time_sub['time'], 'train': df_time_train['time']}
        dic_pred_sub[cities[i]] = learning_reg().var_predict_sbs(train, x_sub, df_time, lag = 4, verbose = 1)
        dic_pred_sub[cities[i]] = dic_pred_sub[cities[i]].ewm(span = average).mean()
        prep_sub.scale_back_models(dic_pred_sub[cities[i]])
        dic_pred_sub[cities[i]] = dic_pred_sub[cities[i]] - dic_pred_sub[cities[i]].min()

    df_sj = pd.DataFrame(dic_pred_sub['sj'])
    df_iq = pd.DataFrame(dic_pred_sub['iq'])
    df_pred = pd.concat([df_sj, df_iq], axis=0)
    df_pred = df_pred.reset_index(drop = True)

    df = pd.read_csv(filename3)
    df.drop(['total_cases'], inplace=True, axis=1)
    df['total_cases'] = df_pred['VAR'].astype(int)
    st.write(df)
    df.to_csv('sottomissioni/submission_var_new.csv', sep=',', index = False)



#  ██████  ██████  ███    ██ ███████ ██████   ██████  ███    ██ ████████  ██████
# ██      ██    ██ ████   ██ ██      ██   ██ ██    ██ ████   ██    ██    ██    ██
# ██      ██    ██ ██ ██  ██ █████   ██████  ██    ██ ██ ██  ██    ██    ██    ██
# ██      ██    ██ ██  ██ ██ ██      ██   ██ ██    ██ ██  ██ ██    ██    ██    ██
#  ██████  ██████  ██   ████ ██      ██   ██  ██████  ██   ████    ██     ██████

name1 = 'sottomissioni/submission_var_new'
name2 = 'sottomissioni/submission_var_old_dnn_average_on_20'

df1 = pd.read_csv(name1+'.csv')
df2 = pd.read_csv(name2+'.csv')

st.write(df1)

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

# df2_sj['total_cases'] = df2_sj['total_cases']-df2_sj['total_cases'].min()
# df2_iq['total_cases'] = df2_iq['total_cases']-df2_iq['total_cases'].min()

# df_pred = pd.concat([df2_sj, df2_iq], axis=0)
# df_pred = df_pred.reset_index(drop = True)
# st.write(df_pred)
# df_pred.to_csv('submission2.csv', sep=',', index = False)
