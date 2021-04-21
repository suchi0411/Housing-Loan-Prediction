#!/usr/bin/env python
# coding: utf-8

# # Housing Loan by Finance Company - Loan Predictor

# # Import Modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading Dataset

# In[2]:


df = pd.read_csv('D://Loan Dataset.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# # Preprocessing the Dataset

# In[5]:


# find the null values
df.isnull().sum()


# In[6]:


#fill the missing values from numerical terms - mean
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

#fill the missing values from categorical terms - mode
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])


# In[7]:


df.isnull().sum()


# # Exploratory Data Analysis

# In[8]:


# Categorical Attributes Visualization
sns.countplot(df['Gender'])


# In[9]:


sns.countplot(df['Married'])


# In[10]:


sns.countplot(df['Dependents'])


# In[11]:


sns.countplot(df['Education'])


# In[12]:


sns.countplot(df['Self_Employed'])


# In[13]:


sns.countplot(df['Property_Area'])


# In[14]:


sns.countplot(df['Loan_Status'])


# In[15]:


# Numerical attributes Visualization
sns.distplot(df["ApplicantIncome"])


# # Creation of New Attributes

# In[16]:


# Total Income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# # Log Transformation

# In[17]:


# Apply Log transformation to the attribute
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
sns.distplot(df["ApplicantIncomeLog"])


# In[18]:


df['LoanAmountLog'] = np.log(df['LoanAmount'])
sns.distplot(df['LoanAmountLog'])


# In[19]:


df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'])
sns.distplot(df['Loan_Amount_Term_Log'])


# In[20]:


df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome'])
sns.distplot(df['ApplicantIncomeLog'])


# In[21]:


df['Total_Income_Log'] = np.log(df['Total_Income'])
sns.distplot(df['Total_Income_Log'])


# # Coorelation Matrix

# In[22]:


corr = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot = True, cmap="BuPu")


# In[23]:


# drop unnecessary columns
cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Total_Income','Loan_ID','CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)
df.head()


# # Label Encoding

# In[24]:


from sklearn.preprocessing import LabelEncoder
cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()


# # Train-Test Split

# In[25]:


# Specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
Y = df['Loan_Status']


# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# # Model Training

# In[27]:


# Classify Function
from sklearn.model_selection import cross_val_score
def classify(model, X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test)*100)
    # Cross Validation - it is used for better validation of a model
    # eg: cv-5, train-4, test-1
    score = cross_val_score(model, X, Y, cv=5)
    print("Cross Validation is,",np.mean(score)*100)


# In[28]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, Y)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, Y)


# In[30]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model, X, Y)


# # Confusion Matrix

# In[31]:


model = LogisticRegression()
model.fit(x_train , y_train)


# In[32]:


from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm


# In[33]:


sns.heatmap(cm, annot=True)


# # FINAL APP CODE

# In[34]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

path='D://'
train_dataset='Loan Dataset.csv'

def pred_model(df_train, df_test):
    
    df_train['type']='train'
    
    df_test['type']='predict'
    
    df=pd.concat([df_train,df_test])
    #fill the missing values from numerical terms - mean
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

    #fill the missing values from categorical terms - mode
    df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
    df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
    df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])
    
    np.seterr(divide = 'ignore')
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
    df['LoanAmountLog'] = np.log(df['LoanAmount'])
    df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'])
    df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome'])
    df['Total_Income_Log'] = np.log(df['Total_Income'])
    
    cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term', 'Total_Income','Loan_ID','CoapplicantIncomeLog']
    df = df.drop(columns=cols, axis=1)
    
    from sklearn.preprocessing import LabelEncoder
    cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents']
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X_train = df[df['type']=='train']
    X_train = X_train.drop(columns=['Loan_Status','type'], axis=1)
    
    X_test = df[df['type']=='predict']
    X_test = X_test.drop(columns=['Loan_Status','type'], axis=1)
    
    Y_train = df['Loan_Status'][df['type']=='train']
    Y_test = df['Loan_Status'][df['type']=='predict']
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train , Y_train)
    out = model.predict(X_test)

    return out

'''
#predict dataset inputs
Loan_ID='l1234'
Gender = 'F'
Married = 'No'
Dependents = 1
Education = 'Graduate'
Self_Employed = 'No'
ApplicantIncome = 1
CoapplicantIncome = 1
LoanAmount = 1000
Loan_Amount_Term = 3
Credit_History = 0
Property_Area = 'Semiurban'

mylist=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','ApplicantIncome','CoapplicantIncome',
      'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
df_test = pd.DataFrame([{'Loan_ID':Loan_ID, 'Gender':Gender, 'Married':Married, 'Dependents':Dependents, 'Education':Education, 'Self_Employed':Self_Employed,
                       'ApplicantIncome':ApplicantIncome, 'CoapplicantIncome':CoapplicantIncome, 'Loan_Amount_Term':Loan_Amount_Term, 'Credit_History':Credit_History,
                       'Property_Area':Property_Area}], columns=mylist)

#reading the training dataset
df_train = pd.read_csv(path+train_dataset)

#predicting the loan
pred_model(df_train,df_test)
'''
#creating the app

from tkinter import *
from functools import partial

def operation(Name_text, Married_text, Mobile_No_text, Applicant_Income_text, Loan_ID_text, 
              Any_Dependent_text, Coapplicant_Income_text, Gender_text, Education_text, 
              Self_Employed_text, Loan_Amount_text, Loan_Amount_Term_text, Credit_History_text, Property_Area_text):

    
    name = "Name - " + Name_text.get()
    mob  = "Mobile - " + Mobile_No_text.get()
    married = "Married Status - " + Married_text.get()
    app = "Applicant Income - " + Applicant_Income_text.get()
    loan = "Loan ID - " + Loan_ID_text.get()
    dep = "Dependent - " + Any_Dependent_text.get()
    coapp = "Coapplicant Income - " + Coapplicant_Income_text.get()
    gender = "Gender - " + Gender_text.get()
    edu = "Education - " + Education_text.get()
    selemp = "Self Employed - " + Self_Employed_text.get()
    loanamt = "Loan Amount - " + Loan_Amount_text.get()
    loanamtterm = "Loan Amount Term - " + Loan_Amount_Term_text.get()
    credit = "Credit History - " + Credit_History_text.get()
    proarea = "Property Area - " +  Property_Area_text.get()
    
    n1 = Label(window, text =name).grid(row = 6, column=1)
    n2 = Label(window, text =married).grid(row = 7, column=1)
    n3 = Label(window, text =mob).grid(row = 8, column=1)
    n4 = Label(window, text =app).grid(row = 9, column=1)
    n5 = Label(window, text =loan).grid(row = 10, column=1)
    n6 = Label(window, text =dep).grid(row = 11, column=1)
    n7 = Label(window, text =coapp).grid(row = 12, column=1)
    n8 = Label(window, text =gender).grid(row = 13, column=1)
    n9 = Label(window, text =edu).grid(row = 14, column=1)
    n10 = Label(window, text =selemp).grid(row = 15, column=1)
    n11 = Label(window, text =loanamt).grid(row = 16, column=1)
    n12 = Label(window, text =loanamtterm).grid(row = 17, column=1)
    n13 = Label(window, text =credit).grid(row = 18, column=1)
    n14 = Label(window, text =proarea).grid(row = 19, column=1)
    
    #predict dataset inputs
    Loan_ID = Loan_ID_text.get()
    Gender = Gender_text.get()
    Married = Married_text.get()
    Dependents = Any_Dependent_text.get()
    Education = Education_text.get()
    Self_Employed = Self_Employed_text.get()
    ApplicantIncome = int(Applicant_Income_text.get())
    CoapplicantIncome = float(Coapplicant_Income_text.get())
    LoanAmount = float(Loan_Amount_text.get())
    Loan_Amount_Term = float(Loan_Amount_Term_text.get())
    Credit_History = float(Credit_History_text.get())
    Property_Area = Property_Area_text.get()

    mylist=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','ApplicantIncome','CoapplicantIncome',
          'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
    df_test = pd.DataFrame([{'Loan_ID':Loan_ID, 'Gender':Gender, 'Married':Married, 'Dependents':Dependents, 'Education':Education, 'Self_Employed':Self_Employed,
                           'ApplicantIncome':ApplicantIncome, 'CoapplicantIncome':CoapplicantIncome, 'Loan_Amount_Term':Loan_Amount_Term, 'Credit_History':Credit_History,
                           'Property_Area':Property_Area}], columns=mylist)


    #predicting the loan
    out=pred_model(df_train,df_test)
    print('Out:',out)
    if out==1:
        Label(window,text = "    Yes !!! You are eligible for this loan        ").grid(row=7, column=2)
    if out==0:
        Label(window,text = " Sorry !!! You are not eligible for this loan").grid(row=7, column=2)
    
    return df_test
    
def close():
    window.destroy()
    

window = Tk()
window.geometry("850x600")
window.title("Housing Finance Company")

l1=Label(window,text = "Name").grid(row=0,column=0)
Name_text=StringVar()
e1=Entry(window,textvariable=Name_text).grid(row=0,column=1)

l2=Label(window,text = "Gender*").grid(row=1,column=0)
Gender_text=StringVar()
e2=Entry(window,textvariable=Gender_text).grid(row=1,column=1)

l3=Label(window,text = "Mobile_No").grid(row=2,column=0)
Mobile_No_text=StringVar()
e3=Entry(window,textvariable=Mobile_No_text).grid(row=2,column=1)

l4=Label(window,text = "Loan_ID").grid(row=3,column=0)
Loan_ID_text=StringVar()
e4=Entry(window,textvariable=Loan_ID_text).grid(row=3,column=1)

l5=Label(window,text = "Married*").grid(row=4,column=0)
Married_text=StringVar()
e5=Entry(window,textvariable=Married_text).grid(row=4,column=1)

l6=Label(window,text = "Num_Dependents*").grid(row=0,column=2)
Any_Dependent_text=StringVar()
e6=Entry(window,textvariable=Any_Dependent_text).grid(row=0,column=3)

l7=Label(window,text = "Education*").grid(row=1,column=2)
Education_text=StringVar()
e7=Entry(window,textvariable=Education_text).grid(row=1,column=3)

l8=Label(window,text = "Self_Employed*").grid(row=2,column=2)
Self_Employed_text=StringVar()
e8=Entry(window,textvariable=Self_Employed_text).grid(row=2,column=3)

l9=Label(window,text = "Applicant Income*").grid(row=3,column=2)
Applicant_Income_text=StringVar()
e9=Entry(window,textvariable=Applicant_Income_text).grid(row=3,column=3)

l10=Label(window,text = "Coapplicant Income*").grid(row=4,column=2)
Coapplicant_Income_text=StringVar()
e10=Entry(window,textvariable=Coapplicant_Income_text).grid(row=4,column=3)

l11=Label(window,text = "Property Area*").grid(row=0,column=4)
Property_Area_text=StringVar()
e11=Entry(window,textvariable=Property_Area_text).grid(row=0,column=5)

l12=Label(window,text = "Credit History*").grid(row=1,column=4)
Credit_History_text=StringVar()
e12=Entry(window,textvariable=Credit_History_text).grid(row=1,column=5)

l13=Label(window,text = "Loan Amount*").grid(row=2,column=4)
Loan_Amount_text=StringVar()
e13=Entry(window,textvariable=Loan_Amount_text).grid(row=2,column=5)

l14=Label(window,text = "Loan Amount Term*").grid(row=3,column=4)
Loan_Amount_Term_text=StringVar()
e14=Entry(window,textvariable=Loan_Amount_Term_text).grid(row=3,column=5)

#reading the training dataset
df_train = pd.read_csv(path+train_dataset)
    
operation = partial(operation, Name_text, Married_text, Mobile_No_text, 
                    Applicant_Income_text, Loan_ID_text, Any_Dependent_text, 
                    Coapplicant_Income_text, Gender_text, Education_text, 
                    Self_Employed_text, Loan_Amount_text, Loan_Amount_Term_text,
                    Credit_History_text, Property_Area_text)
df_test = operation
close = partial(close)

b2 = Button(window,text="Check" , command = operation, width=12).grid(row=25,column=2)

b3 = Button(window,text="Close" , command = close, width=12).grid(row=25,column=3)


window.mainloop()


# In[ ]:





# In[ ]:




