setwd("C:/Users/Rahul/Desktop/edwisor")
source("churnout_preprocess.R")

library("C50")
library("caret")
library("DMwR")
library("randomForest")
library("e1071")
library("class")

test_data=read.csv("Test_data.csv",header = TRUE)
test_data=feat_engg(test_data)
if(length(x)!=0){
test_data = subset(test_data, select = names(test_data) != x)
}
for(i in cnames){
  test_data[,i] = (test_data[,i] - min(tdf[,i]))/
    (max(tdf[,i] - min(tdf[,i])))
}

Churn_data_test=subset(test_data, select = Churn)
transformed_data_test=subset(test_data, select = -Churn)
scaled_data_test=cbind(transformed_data_test,Churn_data_test)

accuracy_cal=function(table){
  acc=((table["0","0"]+table["1","1"])/(table["0","0"]+table["1","1"]+table["0","1"]+table["1","0"]))*100
  return(acc)
}

fn_rate_cal=function(table){
  fn=((table["1","0"])/(table["1","1"]+table["1","0"]))*100
  return(fn)
}

cost_cal=function(table){
  cost=table["1","1"]*8-table["0","1"]*2-table["1","0"]*10
  return(cost)
}

newData = SMOTE(Churn ~ ., scaled_data, perc.over = 500)
#C5 decision tree
set.seed(1234)
C50_model = C5.0(Churn ~., newData, trials = 100, rules = FALSE)
C50_Predictions = predict(C50_model, scaled_data_test[,-14], type = "class")
ConfMatrix_C50 = table(scaled_data_test$Churn, C50_Predictions)
accuracy_c5=accuracy_cal(ConfMatrix_C50)
fn_c5=fn_rate_cal(ConfMatrix_C50)
#Random forest
ntrees=c(seq(10,500,10))
accuracy_rf= vector(mode="numeric", length=0)
fn_rf= vector(mode="numeric", length=0)
rfbestestimator=function(){
for(n in ntrees){
  set.seed(1234)
  RF_model = randomForest(Churn ~ ., newData, importance = TRUE, ntree = n)
  RF_Predictions = predict(RF_model, scaled_data_test[,-14])
  ConfMatrix_RF = table(scaled_data_test$Churn, RF_Predictions)
  accuracy_rf=append(accuracy_rf,accuracy_cal(ConfMatrix_RF))
  fn_rf=append(fn_rf,fn_rate_cal(ConfMatrix_RF))
  
}
  plot(ntrees,accuracy_rf,type = "l",col='red',xlab = "no. of trees",ylab = "accuracy",main = "RF Accuuracy Plot",xaxp  = c(10, 500, 49))
  axis(1, at = c(seq(10,500,10)), tck = 1, lty = 2, col = "grey", labels = NA)

  plot(ntrees,fn_rf,type = "l",xlab = "no. of trees",col='red',ylab = "FN Rate",main = "RF FN Rate Plot",xaxp  = c(10, 500, 49))
  axis(1, at = c(seq(10,500,10)), tck = 1, lty = 2, col = "grey", labels = NA)
  
}
rfbestestimator()
#50 is the most optimized number
set.seed(1234)
RF_model = randomForest(Churn ~ ., newData, importance = TRUE, ntree = 50)
RF_Predictions = predict(RF_model, scaled_data_test[,-14])
ConfMatrix_RF = table(scaled_data_test$Churn, RF_Predictions)
rf_accuracy=accuracy_cal(ConfMatrix_RF)
rf_fn=fn_rate_cal(ConfMatrix_RF)

#support vector classifier

svcbestestimator=function(){
  set.seed(1234)
  est=tune(svm, Churn ~ ., data = newData, 
       ranges = list(gamma = c(0.001, 0.01, 0.1, 1), cost = c(0.001, 0.01, 0.1, 1, 10,1000)),
       tunecontrol = tune.control(sampling = "cross",cross=2)
  )
  
  return(est$best.parameters)
}

bestparams=svcbestestimator()
set.seed(1234)
model_svm = svm(Churn ~ . , newData,gamma=bestparams$gamma,cost=bestparams$cost)
SVC_Predictions = predict(model_svm, scaled_data_test[,-14])
ConfMatrix_SVC = table(scaled_data_test$Churn, SVC_Predictions)
accuracy_svc=accuracy_cal(ConfMatrix_SVC)
fn_svc=fn_rate_cal(ConfMatrix_SVC)

#Logistic regression clasifier
set.seed(1234)
logit_model = glm(Churn ~ . , data = newData, family = "binomial")
logit_Predictions = predict(logit_model, newdata = scaled_data_test[,-14], type = "response")
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)
ConfMatrix_LR = table(scaled_data_test$Churn, logit_Predictions)
accuracy_lr=accuracy_cal(ConfMatrix_LR)
fn_lr=fn_rate_cal(ConfMatrix_LR)
#Naive Bayes classifier
set.seed(1234)
NB_model = naiveBayes(Churn ~ . , data = newData)
NB_Predictions = predict(NB_model, scaled_data_test[,-14], type = 'class')
Confmatrix_nb = table(scaled_data_test$Churn,NB_Predictions)
accuracy_nb=accuracy_cal(Confmatrix_nb)
fn_nb=fn_rate_cal(Confmatrix_nb)
#KNN Classifier
nneighbors=c(seq(3,7,2))
accuracy_knn= vector(mode="numeric", length=0)
fn_knn= vector(mode="numeric", length=0)
knnbestestimator=function(){
  for(n in nneighbors){
    set.seed(1234)
    KNN_Predictions = knn(newData[, 1:13], scaled_data_test[, 1:13], newData$Churn, k = n)
    ConfMatrix_KNN = table(scaled_data_test$Churn, KNN_Predictions)
    accuracy_knn=append(accuracy_knn,accuracy_cal(ConfMatrix_KNN))
    fn_knn=append(fn_knn,fn_rate_cal(ConfMatrix_KNN))
    
  }
  plot(nneighbors,accuracy_knn,type = "l",col='red',xlab = "no. of neighbors",ylab = "accuracy",main = "KNN Accuuracy Plot",xaxp  = c(3, 7, 2))
  axis(1, at = c(seq(3,7,2)), tck = 1, lty = 2, col = "grey", labels = NA)
  
  plot(nneighbors,fn_knn,type = "l",xlab = "no. of neighbors",col='red',ylab = "FN Rate",main = "KNN FN Rate Plot",xaxp  = c(3, 7, 2))
  axis(1, at = c(seq(3,7,2)), tck = 1, lty = 2, col = "grey", labels = NA)
  
}

knnbestestimator()
#7 is the most optimized no.
set.seed(1234)
knn_Predictions = knn(newData[, 1:13], scaled_data_test[, 1:13], newData$Churn, k = 7)
ConfMatrix_KNN = table(scaled_data_test$Churn, KNN_Predictions)
accuracy_knn=accuracy_cal(ConfMatrix_KNN)
fn_knn=fn_rate_cal(ConfMatrix_KNN)
tot_accuracy=c(accuracy_c5,rf_accuracy,accuracy_svc,accuracy_lr,accuracy_nb,accuracy_knn)
tot_fn_rate=c(fn_c5,rf_fn,fn_svc,fn_lr,fn_nb,fn_knn)
namesalgo=c("DT","RF","SVC","LR","NB","KNN")
barplot(tot_accuracy,
        main = "Accuracy Performance",
        xlab = "Algorithm",
        ylab = "Accuracy",
        names.arg = namesalgo,
        col = "darkred")
barplot(tot_fn_rate,
        main = "FN Rate Performance",
        xlab = "Algorithm",
        ylab = "FN Rate",
        names.arg = namesalgo,
        col = "blue")
overall_gain_dt=cost_cal(ConfMatrix_C50)
overall_gain_rf=cost_cal(ConfMatrix_RF)
overall_gain_lr=cost_cal(ConfMatrix_LR)
overall_gain_svm=cost_cal(ConfMatrix_SVC)
overall_gain_nb=cost_cal(Confmatrix_nb)
overall_gain_knn=cost_cal(ConfMatrix_KNN)

tot_overall_gain=c(overall_gain_dt,overall_gain_rf,overall_gain_svm,overall_gain_lr,overall_gain_nb,overall_gain_knn)

barplot(tot_overall_gain,
        main = "Cost Optimization",
        xlab = "Algorithm",
        ylab = "Overall gain",
        names.arg = namesalgo,
        col = "red")

save(C50_model, file = "c5model.rda")