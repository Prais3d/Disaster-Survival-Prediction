# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:03:05 2018

@author: Tan
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
train_df=pd.read_csv('train.csv')
gen_set=pd.read_csv('gender_submission.csv')
test_df=pd.read_csv('test.csv')
print(train_df.columns)
print('--'*30)

#summary of the training dataset
print(train_df.describe(include = "all"))
print('nan value in train data')
print(train_df.isnull().sum())
print('--'*30)
print('nan value in test data')
print(test_df.isnull().sum())
#drop cabin as it has excessive missing values
train_df = train_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis = 1)
#ticket is not a useful feature
train_df = train_df.drop(['Ticket'], axis = 1)
test_df = test_df.drop(['Ticket'], axis = 1)
#only train data has embark
#fillna.mode 0 means filling with columns most occurred values 
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
## combine test and train as single to apply some function
all_data=[train_df,test_df]
## create bin for age features
for dataset in all_data:
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
## create bin for fare features
for dataset in all_data:
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,8,15,30,120], labels=['Low_fare','median_fare', 'Average_fare','high_fare'])
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
### for our reference making a copy of both DataSet start working for copy of dataset
traindf=train_df
testdf=test_df
#print(train_df.columns)
all_dat=[traindf,testdf]
for dataset in all_dat:
    drop_column = ['Age','Fare','Name','PassengerId','SibSp','Parch']
    dataset.drop(drop_column, axis=1, inplace = True)
print('--'*30)
print("Selected Features:")
print(traindf.columns) 
traindf = pd.get_dummies(traindf, columns = ["Sex","Age_bin","Embarked","Fare_bin"],
                             prefix=["Sex","Age_type","Em_type","Fare_type"])
testdf = pd.get_dummies(testdf, columns = ["Sex","Age_bin","Embarked","Fare_bin"],
                             prefix=["Sex","Age_type","Em_type","Fare_type"])
print(traindf.columns)
print(testdf.columns)
#print(train_df.columns)
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
x = traindf.drop("Survived",axis=1)#all_features
y = traindf["Survived"]#target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #Standardize features by removing the mean and scaling to unit variance
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#training dataset using logistic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
#print(y_pred)
print('--------------The Accuracy of the Logistic Regression Model----------------------------')
print('The accuracy of the Logistic Regression is',round(accuracy_score(y_test,y_pred)*100,2))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='3.0f',cmap="Blues")
plt.title('Confusion Matrix', y=1.05, size=15)
from sklearn.metrics import roc_curve
TN = cm [0,0]
FP = cm [0,1]
TP = cm [1,1]
FN = cm [1,0]
Sensetivity = TP/(TP+FP)
Specificity = TN/(TN+FP)
#FNR=FN/(FN+TP)
print ('Sensitivity: ',Sensetivity)
print ('Specificity: ',Specificity)

''' Plotting ROC Curve '''
#fpr=FP/(FP+TN)
#tpr=TP/(TP+FN)
from sklearn.metrics import roc_auc_score
logit_roc_auc = roc_auc_score(y_test, classifier.predict(x_test))
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'o--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Decision Tree
import pandas as pd
#from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
decisionTreeClassifier = DecisionTreeClassifier(criterion='entropy')
dTree = decisionTreeClassifier.fit(x_train, y_train)
y_pred = dTree.predict(x_test)
#dotData = tree.export_graphviz(dTree, out_file=None)
#print(dotData)
print('--------------The Accuracy of the Decision Tree Model----------------------------')
acc_dTree = round(accuracy_score(y_test, y_pred) * 100, 2)
print('The accuracy of the Decision Tree is',acc_dTree)
print()
print('.........................End...........................')