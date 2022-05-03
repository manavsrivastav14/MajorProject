from google.colab import files
uploaded = files.upload()

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd
credits=pd.read_csv("FinalDataset.csv")
credits.head()

Loan_Type=input("Loan_Type: ")
sorted = credits.groupby('Loan_Type')
sorted=sorted.get_group(Loan_Type)
pred=sorted.drop(columns=['Bank_Name','Interest_Rate','Loan_Id','Loan_Type','Loan_Amount','Tenure','Margin','Processing_Fee','Rating'])

norm = MinMaxScaler().fit(pred)
X_train_norm = norm.transform(pred)

age=int(input("Age: "))
Gender=int(input("Gender: "))
Annual_Income=int(input("Annual_Income: "))
Monthly_Emi=int(input("Monthly_Emi: "))
Years_of_Credit_History=float(input("Years_of_Credit_History: "))
Credit_Score=int(input("Credit_Score: "))
X_test=np.array([[age,Gender,Annual_Income,Monthly_Emi,Years_of_Credit_History,Credit_Score]])
X_test.reshape(-1, 1)
X_test_norm = norm.transform(X_test)

neighbors = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='cosine')
neighbors.fit(X_train_norm[:8119])

distances, indices = neighbors.kneighbors(X_test_norm)

final=sorted.index[indices.tolist()]
lst=[]
for i in final.tolist():
  lst.append([i+2,credits.loc[i+2,'Bank_Name'],credits.loc[i+2,'Interest_Rate'],credits.loc[i+2,'Rating'],credits.loc[i+2,'Tenure'],credits.loc[i+2,'Margin'],credits.loc[i+2,'Processing_Fee']])

result=[]
for i in lst:
  if i[1] not in result:
    if len(result)<5:
      result.append(i)

val=int(input("Pass the button input here: "))
#values: Interest rate->0,  Rating ->1, Tenure->2, Margin->3, Processing fee->4

def sortby(val):
  result.sort(key = lambda x: x[val+2])
  return result

print(sortby(val))