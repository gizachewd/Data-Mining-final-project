# The link for the data source is:
#Risk Factors for Hospitalization and Death from COVID-19 in Humanitarian Settings - Humanitarian Data Exchange (humdata.org)

# %%------------- Importing the required packages-----------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# ---------- Importing the datasets --------------
df=pd.read_csv('Covid_risk_factors.csv')

# ---------- Data preprocessing ----------------------------------

# look at first few rows
print(df.head())
print(df.columns)
print(df.shape)

# check the structure of the data
print(df.info())

# check the summary of the data
print(df.describe(include='all'))

# Select important features  (Select features that could be potential affect the outcome)
df1 = df.iloc[: ,[1, 5, 7, 18,19, 22, 30, 35, 36, 40 , 55, 12, 11]].copy()

# check types of each column in the new data frame
print(df.columns)
print(df1.dtypes)
#Index(['age_categories', 'anyinfectious', 'bmi_cat', 'fever',
       #'highbloodpressure_enrollment_13080', 'history_chronic_cat',
       #'low_oxygen94_enrollment', 'sex', 'smoke',
       #'symptoms_any', 'test_reason', 'ever_hospitalized', 'deceased'], dtype='object')

# ----------- Identify unique values and value counts of each column ----------

#print(df1['age_categories'].unique()) #['18-44' '45-64' '65+' '< 18']   #count= 4
#print(df1['anyinfectious'].unique()) #['no' nan 'yes']
#print(df1['bmi_cat'].unique())#[nan 'normal weight' 'overweight' 'underweight' 'obesity'] #count=4
#print(df1['fever'].unique())  #['no' 'yes' nan]  #count2
#print(df1['highbloodpressure_enrollment_13080'].unique()) #['no' 'High blood pressure- over 130/80' nan]  #count=2
#print(df1['history_chronic_cat'].unique()) #['1 chronic condition' '0 none' '2+ chronic conditions'] #count3
#print(df1['low_oxygen94_enrollment'].unique())#['Normal oxygen level- above or equal to 94' 'yes' nan] #count2
#print(df1['sex'].unique()) #['male' 'female'] #count2
#print(df1['smoke'].unique())#['yes' 'no' nan] #count2
#print(df1['symptoms_any'].unique())#['no' 'yes']  #count2
#print(df1['test_reason'].unique()) #['Travel' nan 'COVID19 Symptoms' 'Other' 'Known Exposure'] #count4

# -------- Change longer column names to shorter (easier to use)----------

df1.rename(columns= {'highbloodpressure_enrollment_13080':'highbloodpressure',
                       'low_oxygen94_enrollment': 'low oxygen',
                       'ever_hospitalized': 'hospitalized'}, inplace=True)

# --------- check the null values in each column --------
print(df1.isnull().sum())
    #colum 'any infectious' has close to half Nan values, dropped before analysis
    #column 'Studysite_manuscript is not required for hospitalization status outcome, dropped

# drop columns with significant number of Nan
df1=df1.drop('anyinfectious', axis=1)   #223/519 null

# replace categorical data with the most frequent value in that column
df1 = df1.apply(lambda x: x.fillna(x.value_counts().index[0]))

# again check the null values in each column
print(df1.isnull().sum())

# ---------- Replace string values with numerical values -----------

age_cat=df1['age_categories'].unique()
age_num= np.arange(4)
dict_age=dict(zip(age_cat, age_num))
df1['age_categories'] = df1['age_categories'].replace(dict_age)

bmicat=df1['bmi_cat'].unique()
bmi_num= np.arange(4)
dict_bmi=dict(zip(bmicat, bmi_num))
df1['bmi_cat']=df1['bmi_cat'].replace(dict_bmi)

fever_cat=df1['fever'].unique()
fever_num= np.arange(2)
dict_fever=dict(zip(fever_cat, fever_num))
df1['fever']=df1['fever'].replace(dict_fever)

blood_pres=df1['highbloodpressure'].unique()
blood_pres_num= np.arange(2)
dict_blood_press=dict(zip(blood_pres, blood_pres_num))
df1['highbloodpressure']=df1['highbloodpressure'].replace(dict_blood_press)

hist_chronic=df1['history_chronic_cat'].unique()
hist_chronic_num = np.arange(3)
dict_hist_chronic=dict(zip(hist_chronic, hist_chronic_num))
df1['history_chronic_cat']=df1['history_chronic_cat'].replace(dict_hist_chronic)

lowox=df1['low oxygen'].unique()
lowox_num= np.arange(2)
dict_lowox=dict(zip(lowox, lowox_num))
df1['low oxygen']=df1['low oxygen'].replace(dict_lowox)

sex_cat=df1['sex'].unique()
sex_num= np.arange(2)
dict_sex=dict(zip(sex_cat, sex_num))
df1['sex']=df1['sex'].replace(dict_sex)

smoke_cat=df1['smoke'].unique()
smoke_num= np.arange(2)
dict_smoke=dict(zip(smoke_cat, smoke_num))
df1['smoke']=df1['smoke'].replace(dict_smoke)

symptom_cat=df1['symptoms_any'].unique()
symptom_num= np.arange(2)
dict_symptom=dict(zip(symptom_cat, symptom_num))
df1['symptoms_any']=df1['symptoms_any'].replace(dict_symptom)

test_cat=df1['test_reason'].unique()
test_num= np.arange(4)
dict_test=dict(zip(test_cat, test_num))
df1['test_reason']=df1['test_reason'].replace(dict_test)

# ------- replace the outcome values by 0 and 1 ------
   # ['Never hospitalized (Outpatient managed)'= 0
   # ['Ever hospitalized'] = 1

hospitalized_cat=df1['hospitalized'].unique()
hospitalized_num= [0,1]
dict_hosp=dict(zip(hospitalized_cat, hospitalized_num))
df1['hospitalized']=df1['hospitalized'].replace(dict_hosp)

#drop the 'deceased' column, it is another outcome after hospitalization:
df1=df1.drop('deceased', axis=1)
#print(df1.columns)

print(df1.dtypes)

'''



# ------- feature selection ---------------------------
from sklearn.feature_selection import VarianceThreshold
df1 = df1.drop('hospitalized', axis=1)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(dfX)
print(df1.head)
print(df1.dtypes)
'''

# split the dataset and encode the variables
X= df1.drop(['hospitalized'],axis=1)
y=df1['hospitalized']

# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Standardize the Variables
stdsc = StandardScaler()
stdsc.fit(X_train)
X_train = stdsc.transform(X_train)
X_test= stdsc.transform(X_test)


#visualize the data by pairplot to see the distribution
#sns.pairplot(df1)
#plt.savefig('plot.png', dpi=300, bbox_inches='tight')
#plt.show()

# check the distribution of the target
sns.displot(df1['hospitalized'])
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()

# apply models -----
#%%---------------------- Logistic regression --------------------------

# train the model
lm = LogisticRegression()
lm.fit(X_train,y_train)

#perform prediction
lmpred = lm.predict(X_test)

# Metrix Evaluations
print("lmconfusion matrix: ")
print(confusion_matrix(y_test,lmpred))
print("lm Classification Report: ")
print(classification_report(y_test,lmpred))
print("lm Accuracy : ", accuracy_score(y_test, lmpred) * 100)

# #%%------------------------ KNN  -------------------------------------
# train the model
knn = KNeighborsClassifier(n_neighbors=3)

#performing training
knn.fit(X_train,y_train)

#make predictions
knnpred = knn.predict(X_test)

# Metrix Evaluations
print("knn confusion matrix: ")
print(confusion_matrix(y_test,knnpred))
print("knn Classification Report: ")
print(classification_report(y_test,knnpred))
print("knn Accuracy : ", accuracy_score(y_test, knnpred) * 100)

#%%------------------------SVM -----------------------------------------------
# train the model
svm = SVC(kernel='linear', C=1.0, random_state=42)

#performing training
svm.fit(X_train, y_train)

#make predictions
svmpred = svm.predict(X_test)

# Metrix Evaluations
print("SVM confusion matrix: ")
print(confusion_matrix(y_test,svmpred))
print("SVM Classification Report: ")
print(classification_report(y_test,svmpred))
print("SVM Accuracy : ", accuracy_score(y_test, svmpred) * 100)


#------------Decision Tree ---------

dtmodel = DecisionTreeClassifier(criterion='gini')
dtmodel.fit(X_train,y_train)

dtpred = dtmodel.predict(X_test)

print("dt confusion matrix: ")
print(confusion_matrix(y_test,dtpred))
print("dt Classification Report: ")
print(classification_report(y_test,dtpred))
print("dt Accuracy : ", accuracy_score(y_test, dtpred) * 100)

#---------------- Random Forest -----------------

rfclf = RandomForestClassifier(n_estimators=100)

# perform training
rfclf.fit(X_train, y_train)

#make predictions
rfpred = rfclf.predict(X_test)

#metrix
print("rf confusion matrix: ")
print(confusion_matrix(y_test,rfpred))
print("rf Classification Report: ")
print(classification_report(y_test,rfpred))
print("rf Accuracy : ", accuracy_score(y_test, rfpred) * 100)

#--------------------------Naive Bayes -----------------------
# creating the classifier object
NBclf = GaussianNB()

# performing training
NBclf.fit(X_train, y_train)

# make predictions

NB_pred = NBclf.predict(X_test)
NB_pred_score = NBclf.predict_proba(X_test)

# calculate metrics

print("NB confusion matrix: ")
print(confusion_matrix(y_test,NB_pred))
print("NB Classification Report: ")
print(classification_report(y_test, NB_pred))
print("NB Accuracy : ", accuracy_score(y_test, NB_pred) * 100)

# ------ use votting classifier (Hard Voting using accuracy_score from all models ------

voting_clf= VotingClassifier(estimators=[('knn', knn),
                                         ('svm', svm),
                                         ('lm', lm),
                                         ('dtmodel', dtmodel),
                                         ('rfclf', rfclf),
                                         ('NBclf', NBclf)])

voting_clf.fit(X_train, y_train)
voting_clf.score(X_test,y_test)
votclfpred=voting_clf.predict(X_test)

accuracy = accuracy_score(y_test,votclfpred)
print("Accuracy  = {} %".format(accuracy*100))
print(confusion_matrix(y_test, votclfpred))


# ---Hard Voting from three three models ---

voting_clf1= VotingClassifier(estimators=[('svm', svm),
                                         ('logitreg', lm),
                                         ('NBclf', NBclf)])
voting_clf1.fit(X_train, y_train)
voting_clf1.score(X_test,y_test)
votclfpred1=voting_clf1.predict(X_test)

accuracy1 = accuracy_score(y_test,votclfpred1)
print("Accuracy1  = {} %".format(accuracy*100))
print(confusion_matrix(y_test, votclfpred1))


# -------- Apply K-fold methods to optimize models ---------

kf = KFold(n_splits=5, random_state=1, shuffle=True)
kfmodel = voting_clf1
# evaluate model
scores = cross_val_score(kfmodel, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
# report performance
print('Accuracy KF: %.3f (%.3f)' % (mean(scores), std(scores)))


# ------- check correlation between the variables ------
sns.heatmap(df1.corr())
plt.show()

# ------ drop the features with low corellation, creat new data frame2 ---

df2=df1.drop(['highbloodpressure', 'bmi_cat'], axis=1)
print(df2.columns)

# split the dataset and encode the variables
X2= df2.drop(['hospitalized'],axis=1)
y2=df2['hospitalized']

# Splitting the dataset into train and test
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Standardize the Variables
stdsc = StandardScaler()
stdsc.fit(X_train2)
X_train2 = stdsc.transform(X_train2)
X_test2= stdsc.transform(X_test2)

# -------- run voting classifier model on df2 ------

voting_clf2= VotingClassifier(estimators= [('lm', LogisticRegression()),
                                         ('dtmodel', DecisionTreeClassifier()),
                                         ('rfclf', RandomForestClassifier())])

voting_clf2.fit(X_train2, y_train2)
voting_clf2.score(X_test2,y_test2)
votclfpred2=voting_clf2.predict(X_test2)

accuracy2 = accuracy_score(y_test,votclfpred1)
print("Accuracy2 = {} %".format(accuracy*100))
print(confusion_matrix(y_test2, votclfpred2))

# ------- add optimizer for the models ------
#---------------- Random Forest -----------------
        #increase the n_estimator to 200
rfclf2 = RandomForestClassifier(n_estimators=200)
rfclf2.fit(X_train, y_train)
rfpred2 = rfclf.predict(X_test)

#metrix
print("rf2 confusion matrix: ")
print(confusion_matrix(y_test,rfpred))
print("rf2 Classification Report: ")
print(classification_report(y_test,rfpred))
print("rf2 Accuracy : ", accuracy_score(y_test, rfpred2) * 100)


# --------- Grid search to update parameters -----

from sklearn.model_selection import GridSearchCV
param_grid= {'C': [0.1, 1, 10, 100, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid= GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)
#print(grid.best_params_)
grid_predictions=grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test,grid_predictions))

print('#',50*"-")
#%%%% ----------------End--------------------
