
# Import all Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import category_encoders as cat
from sklearn.preprocessing import OneHotEncoder

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


print(x_mod.shape)
print(y_mod.shape)

x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'] = x_mod['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(' EUR', '')
x_mod['Work Experience in Current Job [years]'] = x_mod['Work Experience in Current Job [years]'].str.replace('#NUM!', '0')
# Graph Plot
#X = x_mod.iloc[:,0]
#Y = x_mod.iloc[0:,-1]
#plt.plot(X,Y)

# Modifying the Datatype of columns
#x_mod.astype({'Year of Record': 'int64'}).dtypes

# Dealing with NaN
#x_mod['Year of Record'].fillna(method = 'ffill', inplace = True)
x_mod = x_mod.dropna(subset = ['Year of Record', 'Work Experience in Current Job [years]', 'Satisfation with employer', 'Gender', 'Country', 'University Degree', 'Hair Color','Profession'])
#x_mod = x_mod.drop(['Year of Record'], axis = 1)
#x_mod = x_mod.drop(['Work Experience in Current Job [years]'], axis = 1)

x_mod['Year of Record'].value_counts()
# Dealing with Categorial Data
# Dropping Max Option Columns
x_mod = x_mod.drop(['Country', 'Profession', 'Size of City', 'Yearly Income in addition to Salary (e.g. Rental Income)' ], axis = 1)
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


#np.any(np.isnan(x_mod))
# Label Encoding

#x_mod.isnull().sum()

# Move Target Income column to separate variable 


# Removing Target Income column from Input 
#x_mod = x_mod.drop(columns = 'Total Yearly Income [EUR]')

#x_mod = x_mod.iloc[1:,:]
x_train, x_test, y_train, y_test = train_test_split(x_mod, y_mod, test_size = 0.3)


model = LinearRegression()
print(y_train)
print(y_train.shape)
model.fit(x_train, y_train)

pred = model.predict(x_test)

print(mae(pred, y_test))

