#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR as sr
from sklearn.preprocessing import StandardScaler as sts
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

def sort_dataset(dataset_df):
	#TODO: Implement this function
    asc=dataset_df.sort_values(by="year")
    return asc
def split_dataset(dataset_df):
	#TODO: Implement this function
	X=dataset_df.drop("salary",axis=1)
	y=dataset_df["salary"]
	Y=y*0.001
	X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.01)
	return X_train,X_test,Y_train,Y_test
def extract_numerical_cols(dataset_df):
	#TODO: Implement this function
	ext=dataset_df.loc[:,['age','G','PA','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','HBP','SO','GDP','fly','war']]
	return ext                        
def train_predict_decision_tree(X_train, Y_train, X_test):
	#TODO: Implement this function
                       
	dt_reg=DecisionTreeRegressor()
	dt_reg.fit(X_train,Y_train)
	dt_predicted=dt_reg.predict(X_test)
	return dt_predicted
def train_predict_random_forest(X_train, Y_train, X_test):
	#TODO: Implement this function

	rf_reg=RandomForestRegressor()
	rf_reg.fit(X_train, Y_train)
	rf_predicted=rf_reg.predict(X_test)
	return rf_predicted

def train_predict_svm(X_train, Y_train, X_test):
	#TODO: Implement this function
	sc =sts()
	sc.fit(X_train)       
	scaled_X_train=sc.transform(X_train)
	scaled_X_test=sc.transform(X_test)

	sr.fit(scaled_X_train, Y_train)  
	pipe=make_pipeline(sc.transform(X_train),sr(scaled_X_train, Y_train)) 
	sr.predict(scaled_X_test)
	pipe.fit(X_train,Y_train)
	return pipe
def calculate_RMSE(labels, predictions):
	#TODO: Implement this function
	return np.sqrt(np.mean((predictions-labels)**2))
if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))

