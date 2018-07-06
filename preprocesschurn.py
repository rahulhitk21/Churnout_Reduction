# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 23:18:37 2018

@author: Rahul
"""
import pandas as pd
import os
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
import matplotlib.pyplot as plt

os.chdir("C:/Users/Rahul/Desktop/edwisor")
raw_data=pd.read_csv("Train_data.csv")
missing_val=pd.DataFrame(raw_data.isnull().sum())
collist=['international plan','voice mail plan','Churn']
sns.countplot(x="Churn", data=raw_data)
sns.lmplot('total day charge', 
          'total day minutes', 
          data=raw_data, 
          fit_reg=False )

sns.lmplot('total eve charge', 
           'total eve minutes', 
           data=raw_data, 
           fit_reg=False )

sns.lmplot('total night charge',
           'total night minutes', 
           data=raw_data, 
           fit_reg=False )

sns.lmplot('total intl charge', 
           'total intl minutes', 
           data=raw_data,
          fit_reg=False )


def feat_engg(dataframe):
    dataframe['day_charge/minute']=dataframe['total day charge']/dataframe['total day minutes']
    dataframe['eve_charge/minute']=dataframe['total eve charge']/dataframe['total eve minutes']
    dataframe['night_charge/minute']=dataframe['total night charge']/dataframe['total night minutes']
    dataframe['intl_charge/minute']=dataframe['total intl charge']/dataframe['total intl minutes']
    coltodrop=['state','phone number','area code','total day charge','total day minutes','total eve charge','total eve minutes','total night charge','total night minutes','total intl minutes','total intl charge']
    for i in coltodrop:
        dataframe=dataframe.drop(i,axis=1)
    dataframe['international plan']=dataframe['international plan'].replace([' yes', ' no'], [1, 0])
    dataframe['voice mail plan']=dataframe['voice mail plan'].replace([' yes', ' no'], [1, 0])
    if 'Churn' in dataframe.columns.values:
        dataframe['Churn']=dataframe['Churn'].replace([' True.', ' False.'], [1,0])
 
    for j in dataframe.columns[dataframe.isnull().any()].tolist():
        dataframe[j]=dataframe[j].fillna(0)

    return dataframe
raw_data=feat_engg(raw_data)

def chngtocat(dataframe,coltochange):
    for i in coltochange:
        dataframe[i]=dataframe[i].astype('category')
    return dataframe
    
raw_data=chngtocat(raw_data,collist)

class ChiSquaretest:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None 
        self.chi2 = None 
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _store_chisquare_result(self, colX, alpha):
        k=0
        if self.p>alpha:
            k=1
        return k
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        return self._store_chisquare_result(colX,alpha)
    
cT = ChiSquaretest(raw_data)
unimportantColumns=[]
collist.remove('Churn')
for var in collist:
    if (cT.TestIndependence(colX=var,colY="Churn" )==1): 
        unimportantColumns.append(var)
def removeColumns(listofcolumns,dataframe):
    for i in listofcolumns:
        dataframe=dataframe.drop([i],axis=1) 
    return dataframe
transformedDF=removeColumns(unimportantColumns,raw_data)
numerical_columns=list(transformedDF.columns)
collist.append('Churn')
for col in collist:
    if col not in unimportantColumns:
        numerical_columns.remove(col)  
unimportantColumnsfornum=[]
for var in numerical_columns:
    F, p = stats.f_oneway(transformedDF[var], transformedDF['Churn'])
    if p > 0.05 :
     unimportantColumnsfornum.append(var)
transformedDF=removeColumns(unimportantColumnsfornum,transformedDF)
plt.hist(transformedDF[numerical_columns[5]],bins='auto')
df_corr = transformedDF.loc[:,numerical_columns]
f, ax = plt.subplots(figsize=(9, 6))
corr = df_corr.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
           square=True, ax=ax)
copytdf=transformedDF.copy()
scaler = MinMaxScaler()

def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df

scaled_data=scaleColumns(transformedDF,numerical_columns)
sccollist=list(scaled_data.columns)
sccollist.remove('Churn')
sccollist.append('Churn')
scaled_data = scaled_data[sccollist]
