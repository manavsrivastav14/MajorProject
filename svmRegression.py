import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pickle

data_set = pd.read_csv('New 1.csv')

X=data_set.iloc[:, :-1].values
y=data_set.iloc[:, -1].values
y = y.reshape(len(y),1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train,y_train)
# final=[0,25]
# y_pred=sc_y.inverse_transform([regressor.predict(sc_X.transform([final]))])
# print (y_pred)
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))