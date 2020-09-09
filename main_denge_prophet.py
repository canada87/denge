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


# ██████  ██████   ██████  ██████  ██   ██ ███████ ████████
# ██   ██ ██   ██ ██    ██ ██   ██ ██   ██ ██         ██
# ██████  ██████  ██    ██ ██████  ███████ █████      ██
# ██      ██   ██ ██    ██ ██      ██   ██ ██         ██
# ██      ██   ██  ██████  ██      ██   ██ ███████    ██


# sys.exit()

average = 5

city = 0
prep = preprocess_data(normalize = 'no')
train, test, df_time = prep.training(filename1, filename2,
                                  type ='VAR',
                                  submit = 'train',
                                  master = 'yes',
                                  city = city, rem_feat = 'yes',
                                  additional_feat_rem = 'yes',#keep it yes
                                  adding_feature = 'no',
                                  num_weeks = 0,
                                  perc = 0.3,
                                  month ='no')

st.subheader('Prophet')
predict_matrix = learning_reg().prophet_model_fit(train, test, df_time, city)
df_score = learning_reg().score_models(test['total_cases'], predict_matrix)
st.write(df_score)

# ███████ ██    ██ ██████  ███    ███ ██ ████████     ██    ██  █████  ██████
# ██      ██    ██ ██   ██ ████  ████ ██    ██        ██    ██ ██   ██ ██   ██
# ███████ ██    ██ ██████  ██ ████ ██ ██    ██        ██    ██ ███████ ██████
#      ██ ██    ██ ██   ██ ██  ██  ██ ██    ██         ██  ██  ██   ██ ██   ██
# ███████  ██████  ██████  ██      ██ ██    ██          ████   ██   ██ ██   ██

# prep_sub = preprocess_data(normalize = 'no')
# dic_pred_sub = dict()
#
# cities = ['sj', 'iq']
# for i in range(2):
#     st.title(cities[i])
#
#     train, df_time_train = prep_sub.training(filename1, filename2,
#                                   type ='VAR',
#                                   submit = 'submit',
#                                       master = 'yes',
#                                       city = i, rem_feat = 'yes',
#                                       additional_feat_rem = 'yes',#keep it yes
#                                       adding_feature = 'no',
#                                       num_weeks = 0,
#                                       perc = 0.3,
#                                       month ='no',
#                                       verbose =0)
#
#     x_sub, df_time_sub = prep_sub.training(filename0, filename3,
#                                        master = 'no',
#                                         type ='VAR',
#                                         submit = 'submit',
#                                               city = i, rem_feat = 'yes',
#                                               additional_feat_rem = 'yes',#keep it yes
#                                               adding_feature = 'no',
#                                               num_weeks = 0,
#                                               month ='no',
#                                               verbose = 0)
#
#     x_sub['total_cases'] = 0
#     df_time = {'train': df_time_train['time'], 'test': df_time_sub['time']}
#
#     dic_pred_sub[cities[i]] = learning_reg().prophet_model_fit(train, x_sub, df_time, i)
#     dic_pred_sub[cities[i]] = dic_pred_sub[cities[i]].ewm(span = average).mean()
#     media_train = train['total_cases'].mean()
#     media_sub = dic_pred_sub[cities[i]].mean().tolist()[0]
#     dic_pred_sub[cities[i]] = dic_pred_sub[cities[i]] + (media_train-media_sub)
#     dic_pred_sub[cities[i]] = dic_pred_sub[cities[i]] - dic_pred_sub[cities[i]].min()
#
# df_sj = pd.DataFrame(dic_pred_sub['sj'])
# df_iq = pd.DataFrame(dic_pred_sub['iq'])
# df_pred = pd.concat([df_sj, df_iq], axis=0)
# df_pred = df_pred.reset_index(drop = True)
#
# df = pd.read_csv(filename3)
# df.drop(['total_cases'], inplace=True, axis=1)
# df['total_cases'] = df_pred['Prophet'].astype(int)
# st.write(df)
# df.to_csv('submission_prophet.csv', sep=',', index = False)



#  ██████  ██████  ███    ██ ███████ ██████   ██████  ███    ██ ████████  ██████
# ██      ██    ██ ████   ██ ██      ██   ██ ██    ██ ████   ██    ██    ██    ██
# ██      ██    ██ ██ ██  ██ █████   ██████  ██    ██ ██ ██  ██    ██    ██    ██
# ██      ██    ██ ██  ██ ██ ██      ██   ██ ██    ██ ██  ██ ██    ██    ██    ██
#  ██████  ██████  ██   ████ ██      ██   ██  ██████  ██   ████    ██     ██████

# name1 = 'submission_var_old_dnn_average_on_20'
# # name2 = 'submission2'
# name2 = 'submission_prophet_var_ensamble'
#
# df1 = pd.read_csv(name1+'.csv')
# df2 = pd.read_csv(name2+'.csv')
#
# st.write(df1)
# st.write(df2)
#
# df1_sj = df1[df1['city'] == 'sj']
# df1_iq = df1[df1['city'] == 'iq']
#
# df2_sj = df2[df2['city'] == 'sj']
# df2_iq = df2[df2['city'] == 'iq']
#
# ds().nuova_fig(77)
# df1_sj['total_cases'].plot(label = name1)
# df2_sj['total_cases'].plot(label = name2)
# ds().legenda()
# st.pyplot()
#
# ds().nuova_fig(78)
# df1_iq['total_cases'].plot(label = name1)
# df2_iq['total_cases'].plot(label = name2)
# ds().legenda()
# st.pyplot()

##########################

# df_tot_sj = pd.DataFrame()
# df_tot_iq = pd.DataFrame()
#
# df_tot_sj = df1_sj.copy()
# df_tot_sj['total_cases'] = (df1_sj['total_cases'] + df2_sj['total_cases'])/2
#
# st.write(df_tot_sj)
#
# df_tot_iq = df1_iq.copy()
# df_tot_iq['total_cases'] = (df1_iq['total_cases'] + df2_iq['total_cases'])/2
#
# st.write(df_tot_iq)
#
# df_pred = pd.concat([df_tot_sj, df_tot_iq], axis=0)
# df_pred = df_pred.reset_index(drop = True)
# df_pred['total_cases'] = df_pred['total_cases'].astype(int)
# st.write(df_pred)
# df_pred.to_csv('submission_prophet_var_ensamble.csv', sep=',', index = False)
