import numpy as np
import pandas as pd
#%%
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn import tree
#%% Connect Data
import os
os.chdir ("E:\AI Part2")

df= pd.read_csv("titanic.csv")
#%% feature selection and missing values drop
df2=df[['Survived','Pclass','Sex','Age']]
df3=df2.dropna()
#%% lable encoding
le=LabelEncoder()
df3['Sex']=le.fit_transform(df3['Sex'])

#%% define inputs and lables and X_scaling
X=df3[['Pclass','Sex','Age']]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Y=df3['Survived']
#%%
print("="*60)
print("Class Distribution")
print("="*60)
print (Y.value_counts())
#%%
x_train,x_test,y_train,y_test= train_test_split (X_scaled,Y, test_size=0.3,stratify=Y, random_state=1, shuffle=True)

clf= tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_predect=clf.predict(x_test)
#%%
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predect))
print("\nClassification Report:\n", classification_report(y_test,y_predect))
