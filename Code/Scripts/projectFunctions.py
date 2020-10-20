# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
import os
#import sys
import time
import pandas as pd
import numpy  as np
#import nltk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
#nltk.download('punkt')

#Function to load the data
def loadData(path,filename):
    try:
             files = os.listdir(path)
             for f in files:
                 if f == filename:
                     data = pd.read_csv(os.path.join(path,f))
                     return data
            
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

#Function to explore the data
def exploreData(data):
    try:
           #Total number of records                                  
           rows = data.shape[0]
           cols = data.shape[1]    
          
           # Print the results
           print ("-----------------------------------------------------------------------")
           print ("Total number of records: {}".format(rows))
           print ("Total number of features: {}".format(cols))
           print ("-----------------------------------------------------------------------")
           
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def missingValues(data):
    try:
           # Total missing values
           mis_val = data.isnull().sum()
         
           # Percentage of missing values
           mis_val_percent = 100 * mis_val / len(data)
           
           # Make a table with the results
           mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
           
           # Rename the columns
           mis_val_table_ren_columns = mis_val_table.rename(
           columns = {0 : 'Missing Values', 1 : '% of Total Values'})
           mis_val_table_ren_columns.head(4 )
           # Sort the table by percentage of missing descending
           misVal = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
                   '% of Total Values', ascending=False).round(1)
                     
           return misVal, mis_val_table_ren_columns

    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def transformData(df):
    try:    
        # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
        #features_final = pd.get_dummies(features_log_minmax_transform)
        
        features_log_minmax_transform = pd.DataFrame(data = df)

#        scaler = MinMaxScaler() # default=(0, 1)
#        numerical = df.columns
        features_log_minmax_transform = features_log_minmax_transform.apply(lambda x: np.log(x + 1)) 
#        features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_minmax_transform[numerical])
        
#        features_f = features_log_minmax_transform
#        features_f = features_f[~features_f.isin([np.nan, np.inf, -np.inf]).any(1)]
#        ind = np.where(target_f >= np.finfo(np.float64).max)
#        features_log_minmax_transform = np.log(df)
        return features_log_minmax_transform
        
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

#split the data in to train and test data
def splitData(features, testsize):
    try:
        # Split the 'features' and 'income' data into training and testing sets
        X_train, X_test = train_test_split(features,
                                           test_size = testsize, 
                                           random_state = 1)

        # Show the results of the split
        print ("Features training set has {} samples.".format(X_train.shape[0]))
        print ("Features testing set has {} samples.".format(X_test.shape[0]))
        print ("-----------------------------------------------------------------------")
        return X_train, X_test
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def corrPlot(corr):
    try:
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    except Exception as ex:
        print ("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print (message)
        
def gridSearch(X_train, X_test, y_train, y_test,clf):
    try:
        params = {}
         
        scoring_fnc = make_scorer(r2_score)
        learner = GridSearchCV(clf,params,scoring=scoring_fnc)
        results = {}
         
        start_time = time.clock()
        grid = learner.fit(X_train,y_train)
         
        end_time = time.clock()
        results['train_time'] = end_time - start_time
        clf_fit_train = grid.best_estimator_
        start_time = time.clock()
         
        clf_predict_train = clf_fit_train.predict(X_train)
        clf_predict_test = clf_fit_train.predict(X_test)
         
        end_time = time.clock()
        results['pred_time'] = end_time - start_time  
         
        results['acc_train'] = r2_score(y_train, clf_predict_train)
        results['acc_test']  = r2_score(y_test, clf_predict_test)
        
        return results,clf_fit_train      
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def bayesGauMix(data,comp):
    try:
        bgm = BayesianGaussianMixture(n_components=comp,n_init=20)
        bgm.fit(data)
        weights = np.round(bgm.weights_,2)
        clusters = 0
        for i in weights:
            if i > 0:
                clusters = clusters + 1
        
        return clusters                

    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)
           
def gaussMix(data,clust,init):
    try:
        gm = GaussianMixture(n_components=clust,n_init=init)
        clf = gm.fit(data)
        
        return clf                

    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def isoForest(data):
    try:
        ifo = IsolationForest(random_state=42)
        clf = ifo.fit(data)
        
        return clf                

    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

def locOutlier(data,neigbh):
    try:
        lof = LocalOutlierFactor(n_neighbors=neigbh,novelty=False)
        clf = lof.fit(data)
        
        return clf                

    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)