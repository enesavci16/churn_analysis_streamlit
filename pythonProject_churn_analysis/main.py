import streamlit as st

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, auc
from PIL import Image
#Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, auc
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
#DATA
data_churn=pd.read_csv("churn.csv")

image = Image.open('akademi.jpeg')
st.image(image, caption='Istanbul Data Science Academy')

st.title('CHURN PREDICTION')

data = data_churn
data.drop(columns=['RowNumber','CustomerId', 'Surname'], axis=1, inplace=True)
Geography_dum = pd.get_dummies(data["Geography"], drop_first=True)
Gender_dum = pd.get_dummies(data["Gender"], drop_first=True)
data = pd.concat([data, Gender_dum, Geography_dum], axis=1)
data.drop(["Geography", "Gender"], inplace=True, axis=1)
data["AgeBin"] = pd.cut(data["Age"], 5)
data["BalanceBin"] = pd.cut(data["Balance"], 5)
data["CreditScoreBin"] = pd.cut(data["CreditScore"], 5)
data["EstimatedSalaryBin"] = pd.cut(data["EstimatedSalary"], 5)
le = LabelEncoder()
data["AgeCode"] = le.fit_transform(data["AgeBin"])
data["BalanceCode"] = le.fit_transform(data["BalanceBin"])
data["CreditScoreCode"] = le.fit_transform(data["CreditScoreBin"])
data["EstimatedSalaryCode"] = le.fit_transform(data["EstimatedSalaryBin"])
data.drop(["AgeBin", "BalanceBin", "CreditScoreBin", "EstimatedSalaryBin", "CreditScore", "Age", "Balance",
             "EstimatedSalary"], inplace=True, axis=1)

X = data.drop(columns=['Exited'])
y = data['Exited']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
pred=model.predict(X_test)
acc_rf=accuracy_score(pred,y_test)
st.write("Model acuraccy score:")
st.write(acc_rf)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

st.sidebar.header('INPUT')
def user_input_features():
    selected_CreditScore = st.sidebar.slider('Kredi skorunu giriniz', 0,900, 1)
    selected_Geography = st.sidebar.selectbox("Bölge seçiniz", ["France", "Spain", "Germany"])
    selected_Gender = st.sidebar.selectbox("Cinsiyet seçiniz",  ["Male", "Female"])
    selected_Age = st.sidebar.slider( "Yaşınızı giriniz:",18,110)

    selected_Tenure = st.sidebar.slider("Kaç yıldır müşterimiz siniz?", 0,50,1)
    selected_Balance = st.sidebar.number_input("Lütfen bankamızdaki bakiyenizi giriniz:",
                                               value=0,
                                               min_value=0,
                                               max_value=300000)
    selected_NumOfProducts = st.sidebar.slider("Lütfen bankamızdaki ürün sayısını giriniz:",0,10,1)
    selected_HasCrCard = st.sidebar.slider("Sahip olunan kredi kart sayısı", 0,10, 1)
    selected_IsActiveMember = st.sidebar.selectbox("Aktif üye mi?",["Aktif", "Pasif"])
    selected_EstimatedSalary = st.sidebar.number_input("Tahmini gelir seviyesi:",
                                                       value=0,
                                                       min_value=0,
                                                       max_value=200000)


    data = {
        'CreditScore': selected_CreditScore,
        'Geography': selected_Geography,
        'Gender': (selected_Gender),
        'Age': selected_Age,
        'Balance':selected_Balance,
        'Tenure': selected_Tenure,
        'NumOfProducts': selected_NumOfProducts,
        'HasCrCard': selected_HasCrCard,
        'IsActiveMember': selected_IsActiveMember,
        'EstimatedSalary': selected_EstimatedSalary}
    features = pd.DataFrame(data, index=[0])

    return features

#Seçilen Değerleri DF yapıp göster
input_df = user_input_features()
st.header('User Choices')
st.write(input_df)


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

def dummy_geo(df):
    if df["Geography"].values=="France":
       df["Germany"]=0
       df["Spain"] = 0
    elif df["Geography"].values=="Spain":
       df["Germany"]=0
       df["Spain"]=1
    elif df["Geography"].values=="Germany":
       df["Germany"]=1
       df["Spain"]=0
    df.drop(["Geography"],axis=1,inplace=True)
    return df

def dummy_age(df):
    if df["Age"].values > 17.959 and df["Age"].values <= 26.2:
        df["Age"] = df["Age"] = 0
    elif df["Age"].values > 26.2 and df["Age"].values <= 34.4:
            df["Age"] = df["Age"] = 1
    elif df["Age"].values > 34.4 and df["Age"].values <= 42.6:
            df["Age"] = df["Age"] = 2
    elif df["Age"].values > 42.6 and df["Age"].values <= 50.8:
            df["Age"] = df["Age"] = 3
    elif df["Age"].values > 50.8 and df["Age"].values <= 110:
            df["Age"] = df["Age"] = 4
    return df

def dummy_balance(df):
    if df["Balance"].values > 0 and df["Balance"].values <= 50179.618:
        df["Balance"] = df["Balance"] = 0
    elif df["Balance"].values > 50179.618 and df["Balance"].values <= 100359.236:
            df["Balance"] = df["Balance"] = 1
    elif df["Balance"].values > 100359.236 and df["Balance"].values <= 150538.854:
            df["Balance"] = df["Balance"] = 2
    elif df["Balance"].values > 150538.854 and df["Balance"].values <= 200718.472:
            df["Balance"] = df["Balance"] = 3
    elif df["Balance"].values > 200718.472 and df["Balance"].values <= 50000.000:
        df["Balance"] = df["Balance"] = 4
    return df

def dummy_score(df):
    if df["CreditScore"].values > 0 and df["CreditScore"].values <= 450.0:
        df["CreditScore"] = df["CreditScore"] = 0
    elif df["CreditScore"].values > 450.0 and df["CreditScore"].values <= 550.0:
        df["CreditScore"] = df["CreditScore"] = 1
    elif df["CreditScore"].values > 550.0 and df["CreditScore"].values <= 650.0:
        df["CreditScore"] = df["CreditScore"] = 2
    elif df["CreditScore"].values > 650.0 and df["CreditScore"].values <= 750.0:
        df["CreditScore"] = df["CreditScore"] = 3
    elif df["CreditScore"].values > 750.0 and df["CreditScore"].values <= 900:
        df["CreditScore"] = df["CreditScore"] = 4
        return df

def dummy_salary(df):
    if df["EstimatedSalary"].values > 0 and df["EstimatedSalary"].values <= 40007.76:
        df["EstimatedSalary"] = df["EstimatedSalary"] = 0
    elif df["EstimatedSalary"].values > 40007.76 and df["CreditScore"].values <= 80003.94:
        df["EstimatedSalary"] = df["EstimatedSalary"] = 1
    elif df["EstimatedSalary"].values > 80003.94 and df["EstimatedSalary"].values <= 120000.12:
        df["EstimatedSalary"] = df["EstimatedSalary"] = 2
    elif df["EstimatedSalary"].values > 120000.12 and df["EstimatedSalary"].values <= 159996.3:
        df["EstimatedSalary"] = df["EstimatedSalary"] = 3
    elif df["EstimatedSalary"].values > 159996.3 and df["EstimatedSalary"].values <=500000.0 :
        df["EstimatedSalary"] = df["EstimatedSalary"] = 4
        return df

def dummy_gender(df):
    if df["Gender"].values == "Male":
        df["Gender"] = df["Gender"] = 1
    else :
      df["Gender"] = df["Gender"] = 0
    return df

def dummy_active(df):
    if df["IsActiveMember"].values == "Aktif":
        df["IsActiveMember"] = df["IsActiveMember"] = 1
    elif df["IsActiveMember"].values == "Pasif":
      df["IsActiveMember"] = df["IsActiveMember"] = 0
    return df



dummy_geo(input_df)
dummy_age(input_df)
dummy_balance(input_df)
dummy_score(input_df)
dummy_salary(input_df)
dummy_gender(input_df)
dummy_active(input_df)

st.write(input_df)

def predict():

    arr = np.array(input_df) # Convert to numpy array
    #arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    result = loaded_model.predict(input_df)
    return result # Return the prediction



onay = Image.open('onay.png')

ret = Image.open('ret_2.png')

hesapla=st.button("CHURN ANALYSIS")
if hesapla:
    pr=predict()
    if pr==1:
        st.write("CHURN")
        st.image(ret, caption='CHURN')
    elif pr==0:
        print("NOT CHURN")
        st.image(onay, caption='NOT CHURN')





























