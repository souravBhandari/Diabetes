import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV

data = './diabetes.csv'
df = pd.read_csv(data)
print(df.shape)

print(df.head())
print(df.isnull().sum())
#check for unbalance 
print(df.Outcome.value_counts())
X = df.drop('Outcome',axis=1) # predictor feature coloumns
y = df.Outcome


X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)

print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))
# 

sm =SMOTE(random_state=42)
X_res_OS , Y_res_OS = sm.fit_resample(X,y)
print(pd.Series(Y_res_OS).value_counts())

X_train , X_test , y_train , y_test = train_test_split(X_res_OS, Y_res_OS, test_size = 0.20, random_state = 10)

print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))

grid ={
            'n_estimators': [100, 500, 1000,1500, 2000],
            'max_depth' :[2,3,4,5,6,7],
    	    'learning_rate': [0.01,0.1,0.01]
           
        }

m1 = GridSearchCV(GradientBoostingClassifier(), grid, cv=5)
m1.fit(X_train, y_train) 
print(m1.best_params_)
pred = m1.predict(X_test)
print(classification_report(y_test, pred))
print('Accuracy Score : ' + str(accuracy_score(y_test,pred)))

grid ={
            'n_estimators': [100, 500, 1000,1500, 2000]
           
        }
m2 = GridSearchCV(RandomForestClassifier(), grid,cv=5)
m2.fit(X_train, y_train)
print(m2.best_params_)
pred2 = m2.predict(X_test)
print(classification_report(y_test, pred2))
print('Accuracy Score : ' + str(accuracy_score(y_test,pred2)))
