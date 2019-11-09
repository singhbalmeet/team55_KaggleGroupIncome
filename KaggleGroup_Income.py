
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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Read the Training Data File
trainCol = pd.read_csv('tcd-ml-1920-group-income-train.csv')

# Drop least important fields
#trainCol = trainCol.drop(['Size of City','Wears Glasses','Body Height [cm]'], axis = 1)

# Check Information of the Dataframe
#trainCol.info()

# Read the Test File
#testCol = pd.read_csv('tcd-ml-1920-group-income-test.csv')

# Assign Training file to another variable where changes will be made
x_mod = trainCol


#print(x_mod.shape)
#print(y_mod.shape)

x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(' EUR', '')
x_mod['Work Experience in Current Job [years]'] = x_mod['Work Experience in Current Job [years]'].str.replace('#NUM!', '0')
#x_mod['Year of Record'] = x_mod['Year of Record'].replace('#N/A', '1990')
     
     
def zero_hs(hs1):
    if hs1 == '0' or hs1 == 0:
        hs1 = '0'
    return hs1

x_mod['Housing Situation'] = x_mod['Housing Situation'].apply(zero_hs)

def zero_gender(gen1):
    if gen1 == '0' or gen1 == 0:
        gen1 = 'unknown'
    return gen1

x_mod['Gender'] = x_mod['Gender'].apply(zero_gender)

def female_gender(gen2):
    if gen2 == 'f':
        gen2 = 'female'
    return gen2

x_mod['Gender'] = x_mod['Gender'].apply(female_gender)

def small_size(sc1):
    if sc1 <= 10000000:
        sc1 = 'Small City'    
    else:
        sc1 = 'Big City'
    return sc1

x_mod['Size of City'] = x_mod['Size of City'].apply(small_size)


#x_mod['Size of City'] = np.log(x_mod['Size of City'])



# Graph Plot
#X = x_mod['Size of City']
#Y = x_mod.iloc[0:,-1]
#plt.plot(X,Y,".")

# Modifying the Datatype of columns
#x_mod.astype({'Year of Record': 'int64'}).dtypes

# Dealing with NaN
#x_mod = x_mod.dropna(subset = ['Work Experience in Current Job [years]', 'Satisfation with employer', 'Country', 'University Degree', 'Hair Color','Profession'])
x_mod['Year of Record'].fillna(method = 'ffill', inplace = True) 
#x_mod['Year of Record'].fillna(x_mod['Year of Record'].median(), inplace = True)
x_mod['Gender'].fillna(method = 'ffill', inplace = True) 
x_mod['Year of Record'].fillna(method = 'ffill', inplace = True) 
x_mod['Satisfation with employer'].fillna(method = 'ffill', inplace = True) 
x_mod['Country'].fillna(method = 'ffill', inplace = True) 
x_mod['University Degree'].fillna(method = 'ffill', inplace = True) 
x_mod['Hair Color'].fillna(method = 'ffill', inplace = True) 
x_mod['Profession'].fillna(method = 'ffill', inplace = True) 

#x_mod = x_mod.drop(['Year of Record'], axis = 1)
#x_mod = x_mod.drop(['Work Experience in Current Job [years]'], axis = 1)

x_mod['Housing Situation'].value_counts()
# Dealing with Categorial Data
# Dropping Max Option Columns
x_mod = x_mod.drop(['Instance', 'Work Experience in Current Job [years]', 'Wears Glasses'], axis = 1)
y_mod = x_mod.iloc[:,-1]
x_mod = x_mod.drop(columns = 'Total Yearly Income [EUR]')

# Get Dummies
dummy = pd.get_dummies(x_mod['Housing Situation'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['Housing Situation'],axis = 1) 

dummy = pd.get_dummies(x_mod['Satisfation with employer'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['Satisfation with employer'],axis = 1)

dummy = pd.get_dummies(x_mod['Gender'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['Gender'],axis = 1)

dummy = pd.get_dummies(x_mod['University Degree'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['University Degree'],axis = 1)

dummy = pd.get_dummies(x_mod['Hair Color'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['Hair Color'],axis = 1)

dummy = pd.get_dummies(x_mod['Size of City'])
x_mod = pd.concat([x_mod,dummy],axis = 1)
x_mod = x_mod.drop(['Size of City'],axis = 1)

# Label Encoding
x_mod['Country'] = x_mod['Country'].astype('category').cat.codes
x_mod['Profession'] = x_mod['Profession'].astype('category').cat.codes
#target_column = "Housing Situation"
#label_encoder_class = preprocessing.LabelEncoder()#encoding the target variables
#label_encoder = label_encoder_class.fit_transform(x_mod[target_column])

#x_mod.isnull().sum()

# Move Target Income column to separate variable 

#x_mod['Size of City'].value_counts()
# Removing Target Income column from Input 
#x_mod = x_mod.drop(columns = 'Total Yearly Income [EUR]')

#x_mod = x_mod.iloc[1:,:]
x_train, x_t, y_train, y_t = train_test_split(x_mod, y_mod, test_size = 0.3)

model = RandomForestRegressor(max_depth = 20, n_estimators = 10)

#model = LinearRegression()
print(y_train)
print(y_train.shape)
model.fit(x_train, y_train)

pred = model.predict(x_t)

print(mae(pred, y_t))
# 32896.30037442671
# 27643.150071048076
# 27267.476748200348
# 26817.716739730917 (Country Label Encoding)
# 34085 (drop Size)
# 34110 (Not Drop)
# 

testCol = pd.read_csv('tcd-ml-1920-group-income-test.csv')

x_test = testCol

x_test['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_test['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(' EUR', '')
x_test['Work Experience in Current Job [years]'] = x_test['Work Experience in Current Job [years]'].str.replace('#NUM!', ' ')
#x_test['Housing Situation'] = x_test['Housing Situation'].str.replace('0', 'nA')

def zero_hs(hs1):
    if hs1 == '0' or hs1 == 0:
        hs1 = '0'
    return hs1

x_test['Housing Situation'] = x_test['Housing Situation'].apply(zero_hs)

def zero_gender(gen1):
    if gen1 == '0' or gen1 == 0:
        gen1 = 'unknown'
    return gen1

x_test['Gender'] = x_test['Gender'].apply(zero_gender)

def female_gender(gen2):
    if gen2 == 'f':
        gen2 = 'female'
    return gen2

x_test['Gender'] = x_test['Gender'].apply(female_gender)

def small_size(sc1):
    if sc1 <= 10000000:
        sc1 = 'Small City'    
    else:
        sc1 = 'Big City'
    return sc1

x_test['Size of City'] = x_test['Size of City'].apply(small_size)
      
x_test['Year of Record'].fillna(method = 'ffill', inplace = True)
x_test['Size of City'].fillna(method = 'ffill', inplace = True)
x_test['Yearly Income in addition to Salary (e.g. Rental Income)'].fillna(method = 'ffill', inplace = True)
#x_test['Work Experience in Current Job [years]'].fillna(method = 'ffill', inplace = True)
x_test['Satisfation with employer'].fillna(method = 'ffill', inplace = True)
x_test['Gender'].fillna(method = 'ffill', inplace = True)
x_test['Country'].fillna(method = 'ffill', inplace = True)
x_test['University Degree'].fillna(method = 'ffill', inplace = True)
x_test['Hair Color'].fillna(method = 'ffill', inplace = True)
x_test['Profession'].fillna(method = 'ffill', inplace = True)

x_test = x_test.drop(['Instance', 'Total Yearly Income [EUR]','Work Experience in Current Job [years]', 'Wears Glasses'], axis = 1)


# Get Dummies
dummy = pd.get_dummies(x_test['Housing Situation'])
x_test = pd.concat([x_test,dummy],axis = 1)
x_test = x_test.drop(['Housing Situation'],axis = 1) 

dummy = pd.get_dummies(x_test['Satisfation with employer'])
x_test = pd.concat([x_test,dummy],axis = 1)
x_test = x_test.drop(['Satisfation with employer'],axis = 1)

dummy = pd.get_dummies(x_test['Gender'])
x_test = pd.concat([x_test,dummy],axis = 1)
x_test = x_test.drop(['Gender'],axis = 1)

dummy = pd.get_dummies(x_test['University Degree'])
x_test = pd.concat([x_test,dummy],axis = 1)
x_test = x_test.drop(['University Degree'],axis = 1)

dummy = pd.get_dummies(x_test['Hair Color'])
x_test = pd.concat([x_test,dummy],axis = 1)
x_test = x_test.drop(['Hair Color'],axis = 1)

dummy = pd.get_dummies(x_test['Size of City'])
x_test = pd.concat([x_test,dummy],axis = 1)
x_test = x_test.drop(['Size of City'],axis = 1)

x_test['Country'] = x_test['Country'].astype('category').cat.codes
x_test['Profession'] = x_test['Profession'].astype('category').cat.codes

x_test.info()
x_mod.info()

x_test.columns
x_mod.columns
#x_test.isnull().sum()

model2 = RandomForestRegressor(max_depth = 20, n_estimators = 10)

model2.fit(x_mod, y_mod)

pred2 = model2.predict(x_test)

print(pred2)
with open('output2', 'w') as out:
    for i in pred2:
        out.write(str(i))
        out.write('\n')
        
        
        
        
 # To do
# 1. Reduce large Value columns 
# 2. FillNa techniques (Missing, Random, Mean, Median)
# 3. Use of different models      
# 4. Make Negative Incomes as 0
