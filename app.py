from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/input')
def input():
    return render_template('form.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/education')
def education():
    return render_template('Education.html')
@app.route('/automobile')
def automobile():
    return render_template('automobile.html')

@app.route('/home_loan')
def home_loan():
    return render_template('home-loan.html')

@app.route('/contact')
def contact():
    return render_template('contact us.html')

@app.route('/recommend',methods=['POST','GET'])
def prediction():
    model=pickle.load(open('model.pkl','rb'))
    details=[x for x in request.form.values()]
    prediction=[]
    prediction.append(float(details[3]))
    prediction.append(float(details[4]))
    final=[prediction]
    
    ans=sc_y.inverse_transform([model.predict(sc_X.transform(final))])
    
    if ans<794:
        ans=ans-(ans*0.062)

    #recommendation
    age=float(details[0])
    Gender=details[1]
    if(Gender=="Male"):
        Gender=1
    elif(Gender=="Female"):
        Gender=0
    else:
        Gender=0
    Annual_Income=float(details[2])
    Monthly_Emi=float(details[3])
    if(Monthly_Emi>=0.6*(Annual_Income/12)):
        return render_template('error.html')
    Years_of_Credit_History=float(details[4])
    sorting_parameter=details[6]
    Credit_Score=ans
    Loan_type=details[5]
    sorted = data_set.groupby('Loan_Type')
    sorted=sorted.get_group(Loan_type)
    pred=sorted.drop(columns=['Bank_Name','Interest_Rate','Loan_Id','Loan_Type','Loan_Amount','Tenure','Margin','Processing_Fee','Rating'])
    norm = MinMaxScaler().fit(pred)
    X_train_norm = norm.transform(pred)

    X_test=np.array([[age,Gender,Annual_Income,Monthly_Emi,Years_of_Credit_History,Credit_Score]])
    X_test.reshape(-1, 1)
    X_test_norm = norm.transform(X_test)

    neighbors = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='cosine')
    neighbors.fit(X_train_norm[:8119])

    distances, indices = neighbors.kneighbors(X_test_norm)

    final=sorted.index[indices.tolist()]
    lst=[]
    for i in final.tolist():
        lst.append([i+2,data_set.loc[i+2,'Bank_Name'],data_set.loc[i+2,'Interest_Rate'],data_set.loc[i+2,'Rating'],data_set.loc[i+2,'Tenure'],data_set.loc[i+2,'Margin'],data_set.loc[i+2,'Processing_Fee']])

    result=[]

    for i in lst:
        if i[1] not in result:
            if len(result)<5:
                result.append(i)
       
    if(sorting_parameter=="Interest Rate"):
        val=0
    elif(sorting_parameter=="Rating"):
        val=1
    elif(sorting_parameter=="Tenure"):
        val=2
    elif(sorting_parameter=="Margin"):
        val=3
    elif(sorting_parameter=="Processing Fees"):
        val=4
    else:
        val=0

    def sortby(val):
            result.sort(key = lambda x: x[val+2])
            count=1
            for i in result:
                i[0]=count
                count+=1
            return result
    data=sortby(val)
    # print("Data: ",data)
    heading=("Index","Bank","Interest","Rating","Tenure","LTV(Loan To Value)","Processing Fees")
    return render_template('table.html',heading=heading,data=data)


        
        
        
        
        # val=int(input("Pass the button input here: "))
        # #values: Interest rate->0,  Rating ->1, Tenure->2, Margin->3, Processing fee->4

       

if __name__=='__main__':
    data_set = pd.read_csv('FinalDataset.csv')
                                        
    X=data_set.iloc[:, 7:9].values
    y=data_set.iloc[:, 9].values
    y = y.reshape(len(y),1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)

    regressor = SVR(kernel = 'rbf')

    regressor.fit(X_train,y_train)
    pickle.dump(regressor,open('model.pkl','wb'))
    model=pickle.load(open('model.pkl','rb'))

    app.run(debug=True)




