setwd("C:/Users/Rahul/Desktop/edwisor")
library(dplyr)
library("corrgram")

raw_data=read.csv("Train_data.csv",header = TRUE)
is.null(raw_data)
counts = table(raw_data$Churn)
barplot(counts, main="Churnout Distribution",col = 'blue')
attach(raw_data)
plot(total.day.minutes, total.day.charge, main="TDM vs TDC", 
     xlab="Total day Charge ", ylab="Total day minutes ",col='blue')

plot(total.eve.minutes, total.eve.charge, main="TEM vs TEC", 
     xlab="Total eve Charge ", ylab="Total eve minutes ",col='blue')

plot(total.night.minutes, total.night.charge, main="TNM vs TNC", 
     xlab="Total night Charge ", ylab="Total night minutes ",col='blue')

plot(total.intl.minutes, total.intl.charge, main="TIM vs TIC", 
     xlab="Total intl Charge ", ylab="Total intl minutes ",col='blue')

collist=list('international.plan','voice.mail.plan','Churn')
feat_engg=function(dataframe){
  dataframe$day_charge_per_minute=with(dataframe,total.day.charge/total.day.minutes)
  dataframe$eve_charge_per_minute=with(dataframe,total.eve.charge/total.eve.minutes)
  dataframe$night_charge_per_minute=with(dataframe,total.night.charge/total.night.minutes)
  dataframe$intl_charge_per_minute=with(dataframe,total.intl.charge/total.intl.minutes)
  coltodrop=list('state','phone.number','total.day.charge','total.day.minutes','total.eve.charge','total.eve.minutes','total.night.charge','total.night.minutes','total.intl.minutes','total.intl.charge','area.code')
  dataframe=dataframe[ , !(names(dataframe) %in% coltodrop)]
  dataframe$international.plan<-ifelse(dataframe$international.plan==' yes', 1,0)
  dataframe$international.plan = as.factor(as.numeric(dataframe$international.plan))
  dataframe$voice.mail.plan<-ifelse(dataframe$voice.mail.plan==' yes', 1,0)
  dataframe$voice.mail.plan = as.factor(as.numeric(dataframe$voice.mail.plan))
  if("Churn" %in% c(colnames(dataframe))==TRUE){
    dataframe$Churn<-ifelse(dataframe$Churn==' True.', 1,0)
    dataframe$Churn = as.factor(as.numeric(dataframe$Churn))
    }
  dataframe[is.na(dataframe)] = 0
  return(dataframe)
}
raw_data=feat_engg(raw_data)
factor_index = sapply(raw_data,is.factor)
factor_data = raw_data[,factor_index]
x = vector(mode="character", length=0)
for (i in (1:length(names(factor_data))))
{
  a=(chisq.test(table(factor_data$Churn,factor_data[,i])))
  if(a$p.value>0.05)
  {
    x=append(x,names(factor_data)[i])
  }
}
if(length(x)!=0){
  raw_data = subset(raw_data, select = names(raw_data) != x)
}
numeric_index = sapply(raw_data,is.numeric)
numeric_data = raw_data[,numeric_index]
cnames = colnames(numeric_data)
corrgram(raw_data[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
hist(raw_data$total.intl.calls,main="Intl call distribution",col='red',xlab = 'no. of intl calls')
tdf=data.frame(raw_data)
for(i in cnames){
  raw_data[,i] = (raw_data[,i] - min(raw_data[,i]))/
    (max(raw_data[,i] - min(raw_data[,i])))
}

Churn_data=subset(raw_data, select = Churn)
transformed_data=subset(raw_data, select = -Churn)
scaled_data=cbind(transformed_data,Churn_data)



  


