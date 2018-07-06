setwd("C:/Users/Rahul/Desktop/edwisor")
source("churnout_preprocess.R")
library("C50")
print("Excecution started")
args = commandArgs(trailingOnly=TRUE)
predict_raw_data=read.csv(args[1],header = TRUE)
predict_data=feat_engg(predict_raw_data)
if(length(x)!=0){
predict_data = subset(predict_data, select = names(predict_data) != x)
}
for(i in cnames){
  predict_data[,i] = (predict_data[,i] - min(tdf[,i]))/
    (max(tdf[,i] - min(tdf[,i])))
}

c50predict=get(load(file = "c5model.rda"))
Predictions = predict(c50predict,predict_data, type = "class")
churn_df_predict=data.frame(Churn = Predictions)
final_data=cbind(predict_raw_data,churn_df_predict)
final_data$Churn=ifelse(final_data$Churn=='1', "Yes","No")
write.csv(final_data, file = "predict_r_submission.csv",row.names=FALSE)

print("prediction has been made successfully")

