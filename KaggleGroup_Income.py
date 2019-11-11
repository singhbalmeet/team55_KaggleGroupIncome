# Import all Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import category_encoders as cat
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from feature_engine.categorical_encoders import OneHotCategoricalEncoder

fill_col_dict = {'Year of Record': 1999.0,
 'Gender':'other',
 'Age': 15,
 'Profession': 'other',
 'University Degree': 'No',
 'Hair Color': 'Black',
 'Size of City': 'Small City',
 'Work Experience in Current Job [years]': '24'}

def encode(labelled_data, unlabelled_data, columns):
    encoder = OneHotCategoricalEncoder(
        top_categories=None,
        variables=columns,  # we can select which variables to encode
        drop_last=True)
    encoder.fit(labelled_data)
    labelled_data = encoder.transform(labelled_data)
    unlabelled_data = encoder.transform(unlabelled_data)
    return labelled_data, unlabelled_data

def preprocess(x_local):
    x_local = x_local[np.isfinite(x_local["Age"])]
    x_local = x_local[np.isfinite(x_local["Year of Record"])]
    x_local['Total Yearly Income [EUR]'] = x_local['Total Yearly Income [EUR]'].apply(np.log)
#    x_local['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_local['Yearly Income in addition to Salary (e.g. Rental Income)'].apply(np.log)
    return x_local    

def group(x_local,col,threshold):
    counts = x_local[col].value_counts()
    index = counts[counts<=threshold].index
    x_local[col] = x_local[col].replace(index,"other")
    return x_local

def label_encoding(x_local):
    x_local['Country'] = x_local['Country'].astype('category').cat.codes
    x_local['Profession'] = x_local['Profession'].astype('category').cat.codes
    return x_local


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
        gen1 = 'male'
    if gen1 == 'unknown':
        gen1 == 'other'
    return gen1


def female_gender(gen2):
    if gen2 == 'f':
        gen2 = 'female'
    return gen2


def small_size(sc1):
    if sc1 <= 10000000:
        sc1 = 'Small City'
    else:
        sc1 = 'Big City'
    return sc1


def dealing_with_nan(x_local):
    # dropna:
     #x_mod = x_mod.dropna(subset = ['Work Experience in Current Job [years]', 'Satisfation with employer',
    # 'Country', 'University Degree', 'Hair Color','Profession'])
    for col in fill_col_dict.keys():
        x_local[col] = x_local[col].fillna(fill_col_dict[col])
    

    """
    # fillna:
    x_local['Year of Record'].fillna(method='ffill', inplace=True)
    x_local['Size of City'].fillna(method='ffill', inplace=True)
    x_local['Gender'].fillna(method='ffill', inplace=True)
    x_local['Satisfation with employer'].fillna(method='ffill', inplace=True)
    x_local['Country'].fillna(method='ffill', inplace=True)
    x_local['University Degree'].fillna(method='ffill', inplace=True)
    x_local['Hair Color'].fillna(method='ffill', inplace=True)
    x_local['Profession'].fillna(method='ffill', inplace=True)
    """
    x_local['Yearly Income in addition to Salary (e.g. Rental Income)'].fillna(method='ffill', inplace=True)
    return x_local


def drop_unnessacary_columns(x_local):
    x_local = x_local.drop(['Instance', 'Hair Color'], axis=1)
    x_local = x_local.drop(columns='Total Yearly Income [EUR]')
    return x_local


def make_dummies(x_local):
    """
    encoder = OneHotCategoricalEncoder(
        top_categories=None,
        variables=columns,  # we can select which variables to encode
        drop_last=True)
    encoder.fit(x_local)
    x_local['Housing Situation'] = encoder.transform(x_local['Housing Situation'])
    x_local['Satisfaction with employer'] = encoder.transform(x_local['Satisfaction with employer'])
    x_local['Gender'] = encoder.transform(x_local['Gender'])
    x_local['University Degree'] = encoder.transform(x_local['University Degree'])
    x_local['Size of City'] = encoder.transform(x_local['Size of City'])
 #   unlabelled_data = encoder.transform(unlabelled_data)
    
    """
    dummy = pd.get_dummies(x_local['Housing Situation'])
    x_local = pd.concat([x_local, dummy], axis=1)
    x_local = x_local.drop(['Housing Situation'], axis=1)

    dummy = pd.get_dummies(x_local['Satisfation with employer'])
    x_local = pd.concat([x_local, dummy], axis=1)
    x_local = x_local.drop(['Satisfation with employer'], axis=1)

    dummy = pd.get_dummies(x_local['Gender'])
    x_local = pd.concat([x_local, dummy], axis=1)
    x_local = x_local.drop(['Gender'], axis=1)

    dummy = pd.get_dummies(x_local['University Degree'])
    x_local = pd.concat([x_local, dummy], axis=1)
    x_local = x_local.drop(['University Degree'], axis=1)
    
#    dummy = pd.get_dummies(x_local['Country'])
#    x_local = pd.concat([x_local, dummy], axis=1)
#    x_local = x_local.drop(['Country'], axis=1)

#    dummy = pd.get_dummies(x_local['Profession'])
#    x_local = pd.concat([x_local, dummy], axis=1)
#    x_local = x_local.drop(['Profession'], axis=1)
#    dummy = pd.get_dummies(x_local['Hair Color'])
#    x_local = pd.concat([x_local, dummy], axis=1)
#    x_local = x_local.drop(['Hair Color'], axis=1)

    dummy = pd.get_dummies(x_local['Size of City'])
    x_local = pd.concat([x_local, dummy], axis=1)
    x_local = x_local.drop(['Size of City'], axis=1)

    #dummy = pd.get_dummies(x_local['Wears Glasses'])
    #x_local = pd.concat([x_local, dummy], axis=1)
    #x_local = x_local.drop(['Wears Glasses'], axis=1)
    
    return x_local



trainCol = pd.read_csv('tcd-ml-1920-group-income-train.csv')
x_mod = trainCol
#x_mod = preprocess(x_mod)
x_mod['Total Yearly Income [EUR]'] = x_mod['Total Yearly Income [EUR]'].apply(np.log)
x_mod = replace_wrong_strings(x_mod)
#x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(float)
#x_mod = preprocess(x_mod)
x_mod['Housing Situation'] = x_mod['Housing Situation'].apply(zero_hs)
x_mod['Gender'] = x_mod['Gender'].apply(zero_gender)
x_mod['Gender'] = x_mod['Gender'].apply(female_gender)
x_mod['Size of City'] = x_mod['Size of City'].apply(small_size)
x_mod['University Degree'] = x_mod['University Degree'].apply(zero_ud)
x_mod['Country'] = x_mod['Country'].apply(zero_country)
x_mod = x_mod.dropna(subset = ['Year of Record','Work Experience in Current Job [years]', 'Satisfation with employer','Profession','Country','University Degree', 'Hair Color'])
#x_mod = dealing_with_nan(x_mod)
x_mod = group(x_mod,'Country',30)
x_mod = group(x_mod,'Profession',30)
x_mod['Work Experience in Current Job [years]'] = x_mod['Work Experience in Current Job [years]'].astype(np.float)
x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(np.float)
# Necessary to take only positive values for using logs. Otherwise, we get an error.
#x_mod['Total Yearly Income [EUR]'] = x_mod['Total Yearly Income [EUR]'].apply(np.log)
#x_mod = x_mod[x_mod['Total Yearly Income [EUR]'] > 3000]

# Remove outliers
#x_mod = x_mod[x_mod['Total Yearly Income [EUR]'] <= 4000000]
x_mod.info
y_mod = x_mod.iloc[:, -1]
x_mod = drop_unnessacary_columns(x_mod)
x_mod = make_dummies(x_mod)
x_mod = x_mod.rename(columns = {'Work Experience in Current Job [years]': 'Work Experience', 'Body Height [cm]': 'Body Height'})
y_mod = y_mod.rename(columns = {'Total Yearly Income [EUR]': 'Total Yearly Income'})
x_mod = label_encoding(x_mod)
x_mod = preprocessing.MinMaxScaler().fit_transform(x_mod)
x_mod = pd.DataFrame(x_mod)

x_train, x_t, y_train, y_t = train_test_split(x_mod, y_mod, test_size=0.3)
#model = RandomForestRegressor(max_depth=10, n_estimators=15)
#model = LinearRegression()
model = xgb.XGBRegressor(objective = "reg:linear", booster = 'gbtree', random_state = 42)
print(y_train)
print(y_train.shape)
model.fit(x_train, y_train)
pred = model.predict(x_t)
pred = np.exp(pred)
y_t = np.exp(y_t)
print(mae(pred, y_t))
# 32896.30037442671
# 27643.150071048076
# 27267.476748200348
# 26817.716739730917 (Country Label Encoding)
# 34085 (drop Size)
# 34110 (Not Drop)




# ----------------------------- TEST FILE ------------------------ #


testCol = pd.read_csv('tcd-ml-1920-group-income-test.csv')
x_test = testCol
#x_test['Total Yearly Income [EUR]'] = x_test['Total Yearly Income [EUR]'].apply(np.log)
x_test = replace_wrong_strings(x_test)
# x_test['Housing Situation'] = x_test['Housing Situation'].str.replace('0', 'nA')
x_test['Housing Situation'] = x_test['Housing Situation'].apply(zero_hs)
x_test['Gender'] = x_test['Gender'].apply(zero_gender)
x_test['Gender'] = x_test['Gender'].apply(female_gender)
x_test['Size of City'] = x_test['Size of City'].apply(small_size)
x_test['University Degree'] = x_test['University Degree'].apply(zero_ud)
x_test = dealing_with_nan(x_test)
x_test = group(x_test,'Country',30)
x_test = group(x_test,'Profession',30)
x_test['Work Experience in Current Job [years]'] = x_test['Work Experience in Current Job [years]'].astype(np.float)
x_test['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_test['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(np.float)
x_test = drop_unnessacary_columns(x_test)
x_test = make_dummies(x_test)
x_test = x_test.rename(columns = {'Work Experience in Current Job [years]': 'Work Experience', 'Body Height [cm]': 'Body Height'})
x_test = label_encoding(x_test)
#print(x_test.info())
#print(x_mod.info())
#print(x_test.columns)
#print(x_mod.columns)
#model2 = RandomForestRegressor(max_depth=10, n_estimators=15)
#model2 = LinearRegression()
model2 = xgb.XGBRegressor(objective = "reg:linear", booster = 'gbtree', random_state = 42)
model2.fit(x_mod, y_mod)
pred2 = model2.predict(x_test)
pred2 = np.exp(pred2)
print(pred2)
with open('output3', 'w') as out:
    for i in pred2:
        out.write(str(i))
        out.write('\n')

# To do
# 1. Reduce large Value columns
# 2. FillNa techniques (Missing, Random, Mean, Median)
# 3. Use of different models
# 4. Make Negative Incomes as 0
