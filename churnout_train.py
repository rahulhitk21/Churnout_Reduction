# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 18:56:05 2018

@author: Rahul
"""
import joblib
import pandas as pd
import os
import numpy as np
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
os.chdir("C:/Users/Rahul/Desktop/edwisor")
import preprocesschurn as pchurn

test_data=pd.read_csv("Test_data.csv")
test_data=pchurn.feat_engg(test_data)
test_data=pchurn.chngtocat(test_data,pchurn.collist)
test_data=pchurn.removeColumns(pchurn.unimportantColumns,test_data)
test_data=pchurn.removeColumns(pchurn.unimportantColumnsfornum,test_data)
numerical_cols=pchurn.numerical_columns
tdf=pchurn.copytdf
for var in numerical_cols:
    minimum= min(tdf[var])
    maximum=max(tdf[var])
    test_data[var]=(test_data[var]-minimum)/(maximum-minimum)
test_data = test_data[pchurn.sccollist]

scaled_data=pchurn.scaled_data
X_original=scaled_data.drop(['Churn'],axis=1)
scaled_data['Churn']=scaled_data['Churn'].replace([1,0], ['Yes','No'])
Y_original=scaled_data['Churn']
sm = SMOTE(kind='regular')
X_oversampled, y_oversampled= sm.fit_sample(X_original, Y_original)
testing_features=np.array(test_data.drop(['Churn'],axis=1))
test_data['Churn']=test_data['Churn'].replace([1,0], ['Yes','No'])
testing_target=np.array(test_data['Churn'])

def false_nagative_rate(y_actual, y_hat):
    TP = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]=='Yes':
           TP += 1
        if y_hat[i]=='No' and y_actual[i]!=y_hat[i]:
           FN += 1
    FNR=(FN/(FN+TP))*100

    return(FNR)
def performance_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i] == y_hat[i]=='Yes':
            TP += 1
        if y_hat[i] == 'Yes' and y_actual[i] == 'No':
            FP += 1
        if y_hat[i] == y_actual[i] == 'No':
            TN +=1
        if y_hat[i] == 'No' and y_actual[i] == 'Yes':
            FN +=1

    return(TP, FP, TN, FN)
def cost_measure(perform_tup):
    cost=perform_tup[0]*8-perform_tup[1]*2-perform_tup[3]*10
    return cost   
def accuracy(perform_tup):  
    accuracy=(perform_tup[0]+perform_tup[2])/(perform_tup[0]+perform_tup[1]+perform_tup[2]+perform_tup[3])
    return accuracy
#######################################Testing for different model #########################################
#########################################random_forest###################################################
def showbestestimator():
    estimators = np.arange(10, 501, 10)
    score_rf = {}
    false_ng_rf={}
    for n in estimators:
       actual_model=RandomForestClassifier(n_estimators=n,random_state=25)
       actual_model.fit(X_oversampled, y_oversampled)
       testing_predicted=actual_model.predict(testing_features).tolist()
       testing_actual=testing_target.tolist()
       score=actual_model.score(testing_features, testing_target)
       false_ng=false_nagative_rate(testing_actual,testing_predicted)
       score_rf.update({n:score})
       false_ng_rf.update({n:false_ng})
    x=[]
    y=[]
    z=[]
    for k,v in score_rf.items():
        x.append(k)
        y.append(v)
    for k,v in false_ng_rf.items():
        z.append(v)
    plt.figure(figsize=(20,10))
    plt.plot(x, y)
    plt.grid(True)  
    plt.xticks(np.arange(10,501,10))
    plt.xlabel('Number_of_trees')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure(figsize=(20,10))
    plt.plot(x,z)
    plt.grid(True)  
    plt.xticks(np.arange(10,501,10))
    plt.xlabel('Number_of_trees')
    plt.ylabel('False_negative_rate') 
    plt.show() 
showbestestimator()
#380 is the most optimized number in terms of accuracy and false negative rate

final_model=RandomForestClassifier(n_estimators=380,random_state=25)
final_model.fit(X_oversampled, y_oversampled)
testing_predicted=final_model.predict(testing_features).tolist()
testing_actual=testing_target.tolist()
false_ng_rf=false_nagative_rate(testing_actual,testing_predicted)
perform_rf = performance_measure(testing_actual,testing_predicted)
score_rf=accuracy(perform_rf)*100

###################################decision tree classifier################################
clf_DT = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10)
clf_DT.fit(X_oversampled, y_oversampled)
y_pred_DT = clf_DT.predict(testing_features).tolist()
y_actual= testing_target.tolist()
fals_ng_dt=false_nagative_rate(y_actual,y_pred_DT)
perform_dt = performance_measure(y_actual,y_pred_DT)
score_dt=accuracy(perform_dt)*100


############################Logistic Regression Classifier###################################

clf_Log = LogisticRegression()                        
clf_Log.fit(X_oversampled, y_oversampled)
y_pred_Log = clf_Log.predict(testing_features).tolist()
y_actual_log=testing_target.tolist()
fals_ng_log=false_nagative_rate(y_actual_log,y_pred_Log)
perform_lr = performance_measure(y_actual_log,y_pred_Log)
score_log=accuracy(perform_lr)*100


############################Naive bayes Classifier###################################

clf_nb = GaussianNB()                        
clf_nb.fit(X_oversampled, y_oversampled)
y_pred_nb = clf_nb.predict(testing_features).tolist()
y_actual_nb=testing_target.tolist()
fals_ng_nb=false_nagative_rate(y_actual_nb,y_pred_nb)
perform_nb = performance_measure(y_actual_nb,y_pred_nb)
score_nb=accuracy(perform_nb)*100

#######################################SVM#################################################

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10,1000]
    gammas = [0.001, 0.01, 0.1, 1]
    Kernel=['linear','poly','rbf','sigmoid']
    param_grid = {'C': Cs, 'gamma' : gammas,'kernel':Kernel}
    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

best_params=svc_param_selection(X_oversampled, y_oversampled,2)


clf_SVM = SVC(C=best_params.get('C'), gamma=best_params.get('gamma'), kernel=best_params.get('kernel'))


clf_SVM.fit(X_oversampled, y_oversampled)
y_pred_SVM = clf_SVM.predict(testing_features).tolist()
y_actual_svm = testing_target.tolist()
fals_ng_svm=false_nagative_rate(y_actual_svm,y_pred_SVM)
perform_svc = performance_measure(y_actual_svm,y_pred_SVM)
score_svm=accuracy(perform_svc)*100



#################################KNN algorithm#############################################
def showbestestimatorknn():
    estimators = [3,5,7]
    score_knn = {}
    false_ng_knn={}
    for n in estimators:
       actual_model=KNeighborsClassifier(n_neighbors=n)
       actual_model.fit(X_oversampled, y_oversampled)
       testing_predicted=actual_model.predict(testing_features).tolist()
       testing_actual=testing_target.tolist()
       score=actual_model.score(testing_features, testing_target)
       false_ng=false_nagative_rate(testing_actual,testing_predicted)
       score_knn.update({n:score})
       false_ng_knn.update({n:false_ng})
    x=[]
    y=[]
    z=[]
    for k,v in score_knn.items():
        x.append(k)
        y.append(v)
    for k,v in false_ng_knn.items():
        z.append(v)
    plt.figure(figsize=(6,9))
    plt.plot(x, y)
    plt.grid(True)  
    plt.xlabel('Number_of_points')
    plt.ylabel('Accuracy')
    plt.show()
    #plt.plot(x, z)
    plt.figure(figsize=(6,9))
    plt.plot(x,z)
    plt.grid(True)  
    plt.xlabel('Number_of_points')
    plt.ylabel('False_negative_rate') 
    plt.show() 
showbestestimatorknn()
#n=7 is the most optimized number in terms of accuracy and false negative rate
final_model_knn=KNeighborsClassifier(n_neighbors=7)
final_model_knn.fit(X_oversampled, y_oversampled)
testing_predicted_knn=final_model_knn.predict(testing_features).tolist()
testing_actual_knn=testing_target.tolist()
false_ng_knn=false_nagative_rate(testing_actual_knn,testing_predicted_knn)
perform_knn = performance_measure(testing_actual_knn,testing_predicted_knn)
score_knn=accuracy(perform_knn)*100




summary= pd.DataFrame({'Model_name': ["RF","KNN","SVM","LR","DT",'NB'], 'Accuracy': [score_rf,score_knn,score_svm,score_log,score_dt,score_nb], 'False_negative_rate': [false_ng_rf,false_ng_knn,fals_ng_svm,fals_ng_log,fals_ng_dt,fals_ng_nb]})

fig = plt.figure(figsize=(9,6)) 


ax = fig.add_subplot(111) 
ax2 = ax.twinx() 

width = 0.4

summary.Accuracy.plot(kind='bar', color='red', ax=ax, width=width, position=1)
summary.False_negative_rate.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
ax.set_ylabel('Accuracy')
ax.set_xticklabels(summary.Model_name)
ax2.set_ylabel('False_negative_rate')



plt.show()

"""
In case of business scenario, we assume if a customer is going to churn out, company
should reach out to the customer to retain them. So let’s assume reaching out to the customer cost 2$ and
retaining them gains 10$ for each of them, so in case of each true positive company is gaining 8$ and for
each false positive losing 2$. Also in case of false negative company loses 10$ for each customer and for
true negative as no reaching out is there so it is 0$.
"""
overall_gain_lr=cost_measure(perform_lr)
overall_gain_dt=cost_measure(perform_dt)
overall_gain_nb=cost_measure(perform_nb)
overall_gain_rf=cost_measure(perform_rf)
overall_gain_svc=cost_measure(perform_svc)
overall_gain_knn=cost_measure(perform_knn)



objects = ('LR', 'DT','NB','RF','SVC','KNN')
y_pos = np.arange(len(objects))
performance = [overall_gain_lr,overall_gain_dt,overall_gain_nb,overall_gain_rf,overall_gain_svc,overall_gain_knn]
 
plt.bar(y_pos, performance, align='center', alpha=0.5,color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Overall cost')
plt.title('Algorithm performance')
 
plt.show()


##We found overall gain is higher for logistic regression algorithm so we are choosing logistic regression algorithm as our final model

joblib.dump(clf_Log,'logistic.pkl')
 









