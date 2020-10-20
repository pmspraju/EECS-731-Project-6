# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:09:21 2020

@author: pmspr
"""
import os
import pandas as pd

#load functions from 
from projectFunctions import loadData

path = r'C:\Users\pmspr\Documents\HS\MS\Sem 3\EECS 731\Week 8\HW\Git\EECS-731-Project-6\Data'
files = os.listdir(path)
data_l = []
for filename in files:
    data =  loadData(path,filename)
    data_l.append(data)

data = pd.concat(data_l)  
data.to_csv(os.path.join(path,r'cpu.csv'),index=False)
