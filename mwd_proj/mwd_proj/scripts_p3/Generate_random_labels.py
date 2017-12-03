# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 01:52:13 2017

@author: Bharadwaj
"""

import pandas as pd
from random import *

df = pd.read_csv('All_data.csv')
train_data = []
length = len(df.index)

for index, row in df.iterrows():
    if index <= length-10:
        train_data.append([row['movieid'], randint(0, 1)])  
    else:
       break     

df_train_data = pd.DataFrame(data=train_data, index=None, columns=['movieid','label'])

df_train_data.to_csv('movie-label.csv',index=False)
