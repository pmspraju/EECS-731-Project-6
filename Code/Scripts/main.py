# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:14:26 2017

@author: Madhu
"""
# Import libraries necessary for this project
import sklearn
#import networkx as nx
import numpy  as np
import pandas as pd
import seaborn as sns; sns.set()
import datetime as dt
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from pandas import compat

compat.PY3 = True
print ("-----------------------------------------------------------------------")
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#load functions from 
from projectFunctions import loadData, missingValues

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 8\HW\Git\EECS-731-Project-6\Data'
filename = "cpu.csv"
data = loadData(path,filename)

#Check the missing values
misVal, mis_val_table_ren_columns = missingValues(data)
print(mis_val_table_ren_columns.head(20))

##Remove rows with missing target values
#ind = data_raw[data_raw['Date'].isnull()].index.tolist()
#data = data_raw.drop(index=ind, axis=0)

#Remove special characters in the column
#data['target'] = pd.to_numeric(data['target'].str.replace('\(|\)',""),errors='coerce').isnull()
#data['value'] = data['value'].str.replace('\(|\)',"")

#Change the datatypes accordingly
data['value'] = data['value'].astype('float')
data['timestamp'] = pd.to_datetime(data['timestamp'])

#sort the dataframe by timestamp
data = data.sort_values(by='timestamp')
data.set_index(np.arange(len(data.index)))

#compute the elapsed time feature
md = data['timestamp'].min()
ed = md - dt.timedelta(hours=1)
data['elapsed'] = data['timestamp'] - ed 
data['elapsed'] = data['elapsed'] / pd.Timedelta('1 hour')
data['elapsed'] = data['elapsed'].apply(lambda x: round(x,2))

#Compute year,month,day features
data['year'] = data['timestamp'].dt.year
data['year'] = data['year'].astype('int')

data['month'] = data['timestamp'].dt.month
data['month'] = data['month'].astype('int')

data['day'] = data['timestamp'].dt.day
data['day'] = data['day'].astype('int')

#plots
#plt.figure(figsize=(5,5))
#sns.countplot(x='year', data=data)
#plt.title('Year wise observings')
#plt.ylabel('Count', fontsize=12)
#plt.xlabel('Year', fontsize=12)

#Drop rows for year 2013
ind = data[data['timestamp'].dt.year == 2013].index.tolist()
data = data.drop(index=ind, axis=0)

#fig,axes = plt.subplots(1,2,sharex=True, figsize=(16,8))
#lineplot = sns.relplot(x="timestamp", y="value", kind="line", data=data)
#lineplot.fig.autofmt_xdate()

ax = sns.relplot(x="day", y="value", hue="month", data=data);
ax.set(xlim=(0,31))
#Drop the timestamp column as model cannot process
data = data.drop(columns = ['timestamp'], axis=1)

from projectFunctions import exploreData, transformData, splitData
exploreData(data)

data_raw = data
features = transformData(data_raw)
features_s = features
features_s = features_s.drop(columns=['year','elapsed'],axis=1)

X_train, X_test = splitData(features_s,0.3)

from projectFunctions import bayesGauMix, gaussMix, isoForest, locOutlier

#Implement Bayessian gaussian mixture
clusters = bayesGauMix(features_s,20)
#print(clusters)

#Implement Gaussian mixture clustering
gm = gaussMix(X_train,4,10)
weights = np.round(gm.weights_,2)
predict = gm.predict(features_s)
prob    = gm.predict_proba(features_s)
density = gm.score_samples(features_s)
print ("-----------------------------------------------------------------------")
print ("Gaussian mixture converged %s in %d iterations" %(gm.converged_, gm.n_iter_))

data['gclus'] = pd.Series(predict.tolist())
print(data['gclus'].unique())
data['gprob'] = pd.Series(prob.tolist())
data['gdensity'] = pd.Series(density.tolist())

ax = sns.relplot(x="day", y="gdensity", hue="gclus", data=data);
ax.set(xlim=(0,31))

#Implement Isolation Forest
ifo = isoForest(X_train)
predict = ifo.predict(features_s)
density = ifo.score_samples(features_s)

data['iclus'] = pd.Series(predict.tolist())
data['idensity'] = pd.Series(density.tolist())

ax = sns.relplot(x="day", y="value", hue="iclus", data=data);
ax.set(xlim=(0,31))

#Implement local outlier 
lof = locOutlier(X_train,1)
#density = lof.decision_function(features_s)
clus = lof.fit_predict(features_s)

data['lclus'] = pd.Series(clus.tolist())

ax = sns.relplot(x="day", y="value", hue="lclus", data=data);
ax.set(xlim=(0,31))

data.to_csv('test.csv',index=False)