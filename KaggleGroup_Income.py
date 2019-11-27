# Import all Libraries
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
import matplotlib.pyplot as plt
import category_encoders as cat
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# from feature_engine.categorical_encoders import OneHotCategoricalEncoder

d_gender = {}
d_housing = {}
d_satisfaction = {}
d_degree = {}
d_country = {}
d_profession = {}

fill_col_dict = {'Year of Record': 1999.0,
                 'Gender': 'other',
                 'Age': 15,
                 'Profession': 'other',
                 'University Degree': 'No',
                 'Hair Color': 'Black',
                 'Size of City': 'Small City',
                 'Work Experience in Current Job [years]': '24',
                 'Yearly Income in addition to Salary (e.g. Rental Income)':'0'}


def replace_wrong_strings(x_local):
    x_local['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_local[
        'Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(' EUR', '')
    x_local['Work Experience in Current Job [years]'] = x_local['Work Experience in Current Job [years]'].str.replace(
        '#NUM!', '0')
    # x_local['Year of Record'] = x_local['Year of Record'].replace('#N/A', '1990')
    return x_local


def zero_country(country1):
    if country1 == '0' or country1 == 0:
        country1 = 'other'
    return country1


def zero_ud(ud1):
    if ud1 == '0' or ud1 == 0:
        ud1 = 'No'
    return ud1


def zero_hs(hs1):
    if hs1 == '0' or hs1 == 0:
        hs1 = '0'
    return hs1


def zero_gender(gen1):
    if gen1 == '0' or gen1 == 0:
        gen1 = '0'
    return gen1


def female_gender(gen2):
    if gen2 == 'f':
        gen2 = 'female'
    return gen2


def small_size(sc1):
    if sc1 < 100000:
        sc1 = 0
    if sc1 < 9000000:
        sc1 = 1
    else:
        sc1 = 2
    return sc1


def dealing_with_nan(x_local):
    for col in fill_col_dict.keys():
        x_local[col] = x_local[col].fillna(fill_col_dict[col])

    return x_local


def drop_unnessacary_columns(x_local):
    x_local = x_local.drop(['Instance', 'Hair Color'], axis=1)
    x_local = x_local.drop(columns='Total Yearly Income [EUR]')
    return x_local


def make_dummies(x_local, y_local):
    temp = x_local['Housing Situation']
    x_local['Housing Situation'] = TargetEncoder().fit_transform(x_local['Housing Situation'], y_local)
    for i in range(len(temp)):
        try:
            a = temp[i]
        except KeyError:
            a = 'other'
        if a not in d_housing:
            try:
                d_housing[a] = x_local['Housing Situation'][i]
            except KeyError:
                pass

    temp = x_local['Satisfation with employer']
    x_local['Satisfation with employer'] = TargetEncoder().fit_transform(x_local['Satisfation with employer'], y_local)
    for i in range(len(temp)):
        try:
            a = temp[i]
        except KeyError:
            a = 'other'
        if a not in d_satisfaction:
            try:
                d_satisfaction[a] = x_local['Satisfation with employer'][i]
            except KeyError:
                pass

    temp = x_local['Gender']
    x_local['Gender'] = TargetEncoder().fit_transform(x_local['Gender'], y_local)
    for i in range(len(temp)):
        try:
            a = temp[i]
        except KeyError:
            a = 'other'
        if a not in d_gender:
            try:
                d_gender[a] = x_local['Gender'][i]
            except KeyError:
                pass

    temp = x_local['University Degree']
    x_local['University Degree'] = TargetEncoder().fit_transform(x_local['University Degree'], y_local)
    for i in range(len(temp)):
        try:
            a = temp[i]
        except KeyError:
            a = 'other'
        if a not in d_degree:
            try:
                d_degree[a] = x_local['University Degree'][i]
            except KeyError:
                pass

    temp = x_local['Country']
    x_local['Country'] = TargetEncoder().fit_transform(x_local['Country'], y_local)
    for i in range(len(temp)):
        try:
            a = temp[i]
        except KeyError:
            a = 'other'
        if a not in d_country:
            try:
                d_country[a] = x_local['Country'][i]
            except KeyError:
                pass

    temp = x_local['Profession']
    x_local['Profession'] = TargetEncoder().fit_transform(x_local['Profession'], y_local)
    for i in range(len(temp)):
        try:
            a = temp[i]
        except KeyError:
            a = 'other'
        if a not in d_profession:
            try:
                d_profession[a] = x_local['Profession'][i]
            except KeyError:
                pass

    return x_local


def f_country(l):
    try:
        return d_country[l]
    except KeyError:
        return 0

def f_prof(l):
    try:
        return d_profession[l]
    except KeyError:
        return 0


def make_dummies2(x_2):
    x_2['University Degree'] = x_2['University Degree'].apply(lambda l: d_degree[l])
    x_2['Housing Situation'] = x_2['Housing Situation'].apply(lambda l: d_housing[l])
    x_2['Satisfation with employer'] = x_2['Satisfation with employer'].apply(lambda l: d_satisfaction[l])
    x_2['Gender'] = x_2['Gender'].apply(lambda l: d_gender[l])
    x_2['Country'] = x_2['Country'].apply(lambda l: f_country(l))
    x_2['Profession'] = x_2['Profession'].apply(lambda l: f_prof(l))

    return x_2


trainCol = pd.read_csv('tcd-ml-1920-group-income-train.csv')
x_mod = trainCol
x_mod = replace_wrong_strings(x_mod)  # required
x_mod['Housing Situation'] = x_mod['Housing Situation'].apply(zero_hs)
x_mod['Gender'] = x_mod['Gender'].apply(zero_gender)
x_mod['Gender'] = x_mod['Gender'].apply(female_gender)

x_mod['Size of City'] = x_mod['Size of City'].apply(small_size)
x_mod['University Degree'] = x_mod['University Degree'].apply(zero_ud)
x_mod['Country'] = x_mod['Country'].apply(zero_country)

x_mod = dealing_with_nan(x_mod)
x_mod['Work Experience in Current Job [years]'] = x_mod['Work Experience in Current Job [years]'].astype(np.float)
x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_mod[
    'Yearly Income in addition to Salary (e.g. Rental Income)'].astype(np.float)
# Remove outliers
x_mod = x_mod[x_mod['Total Yearly Income [EUR]'] <= 1000000]
############################################################################################
y_mod = x_mod.iloc[:, -1]
############################################################################################
x_mod = drop_unnessacary_columns(x_mod)
x_mod = make_dummies(x_mod, y_mod)

for k in d_degree:
    print(k, d_degree[k])

x_mod = x_mod.rename(
    columns={'Work Experience in Current Job [years]': 'Work Experience', 'Body Height [cm]': 'Body Height'})
y_mod = y_mod.rename(columns={'Total Yearly Income [EUR]': 'Total Yearly Income'})
x_mod = pd.DataFrame(x_mod)

x_train, x_t, y_train, y_t = train_test_split(x_mod, y_mod, test_size=0.3)

# model = xgb.XGBRegressor(objective="reg:linear", booster='gbtree', random_state=42)
trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_t, label=y_t)
params = {
    'max_depth': 20,
    'learning_rate': 0.001,
    "boosting": "gbdt",
    "bagging_seed": 11,
    "metric": 'mse',
    "verbosity": -1,
 }
model = lgb.train(params, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
# model = ensemble.GradientBoostingRegressor(n_estimators=40, max_depth=12, random_state=3)
# model = RandomForestRegressor(max_depth=10, n_estimators=15)
# model = LinearRegression()

# model.fit(x_train, y_train)
pred = model.predict(x_t)
print(mae(pred, y_t))
# 32896.30037442671
# 27643.150071048076
# 27267.476748200348
# 26817.716739730917 (Country Label Encoding)
# 34085 (drop Size)
# 34110 (Not Drop)

# exit(0)
# ----------------------------- TEST FILE ------------------------ #


testCol = pd.read_csv('tcd-ml-1920-group-income-test.csv')
x_test = testCol

x_test = replace_wrong_strings(x_test)
x_test['Housing Situation'] = x_test['Housing Situation'].apply(zero_hs)
x_test['Gender'] = x_test['Gender'].apply(zero_gender)
x_test['Gender'] = x_test['Gender'].apply(female_gender)
x_test['Size of City'] = x_test['Size of City'].apply(small_size)
x_test['University Degree'] = x_test['University Degree'].apply(zero_ud)
x_test['Country'] = x_test['Country'].apply(zero_country)
x_test = dealing_with_nan(x_test)
x_test['Work Experience in Current Job [years]'] = x_test['Work Experience in Current Job [years]'].astype(np.float)
x_test['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_test[
    'Yearly Income in addition to Salary (e.g. Rental Income)'].astype(np.float)
x_test = drop_unnessacary_columns(x_test)
print(x_test['Gender'])
print(x_test['Housing Situation'])
x_test = make_dummies2(x_test)
print(x_test['Gender'])
print(x_test['Housing Situation'])
x_test = x_test.rename(
    columns={'Work Experience in Current Job [years]': 'Work Experience', 'Body Height [cm]': 'Body Height'})
# x_test = label_encoding(x_test)
# print(x_test.info())
# print(x_mod.info())
# print(x_test.columns)
# print(x_mod.columns)
# model2 = RandomForestRegressor(max_depth=10, n_estimators=15)
# model2 = LinearRegression()
# model2 = xgb.XGBRegressor(objective="reg:linear", booster='gbtree', random_state=42)
x_train, x_t, y_train, y_t = train_test_split(x_mod, y_mod, test_size=0.3)
trn_data = lgb.Dataset(x_mod, label=y_mod)
val_data = lgb.Dataset(x_t, label=y_t)
params = {
    'max_depth': 20,
    'learning_rate': 0.001,
    "boosting": "gbdt",
    "bagging_seed": 11,
    "metric": 'mse',
    "verbosity": -1,
}
model2 = lgb.train(params, trn_data, 100000, valid_sets=[trn_data, val_data], verbose_eval=1000,
                   early_stopping_rounds=500)

# model2.fit(x_mod, y_mod)

pred2 = model2.predict(x_test)
# pred2 = np.ex_p(pred2)
print(pred2)
with open('output44', 'w') as out:
    for i in pred2:
        out.write(str(i))
        out.write('\n')

# To do
# 1. Reduce large Value columns
# 2. FillNa techniques (Missing, Random, Mean, Median)
# 3. Use of different models
# 4. Make Negative Incomes as 0
