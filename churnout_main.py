# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 05:35:18 2018

@author: Rahul
"""

import joblib
import os, argparse
import pandas as pd
import numpy as np
os.chdir("C:/Users/Rahul/Desktop/edwisor")
import preprocesschurn as pchurn



class churn(object):
	def getLoadOption(self):
		parser = argparse.ArgumentParser()
		parser.add_argument('--Data_File', action='store', dest='Data_File')
		
		self.result_op = parser.parse_args()
		
		return self.result_op
		
def main():
    print('Execution started')
    cli = churn()
    cli_line= cli.getLoadOption()
    data_file = cli_line.Data_File
    test_data=pd.read_csv(data_file)
    predict_data=test_data.copy()
    test_data=pchurn.feat_engg(test_data)
    pchurn.collist.remove('Churn')
    test_data=pchurn.chngtocat(test_data,pchurn.collist)
    test_data=pchurn.removeColumns(pchurn.unimportantColumns,test_data)
    test_data=pchurn.removeColumns(pchurn.unimportantColumnsfornum,test_data)
    numerical_cols=pchurn.numerical_columns
    tdf=pchurn.copytdf
    for var in numerical_cols:
        minimum= min(tdf[var])
        maximum=max(tdf[var])
        test_data[var]=(test_data[var]-minimum)/(maximum-minimum)
    
    predict_x=np.array(test_data.values)
    model=joblib.load('logistic.pkl')
    predict_y=model.predict(predict_x)
    predict_data['Churn']=predict_y
    predict_data['Churn'].astype(str)
    predict_data.to_csv("Predic_submission.csv",index=False)
    print("prediction has been made successfully")
if __name__ == "__main__":
	main()
    