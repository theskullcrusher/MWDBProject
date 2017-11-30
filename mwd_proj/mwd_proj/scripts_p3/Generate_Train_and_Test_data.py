# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:59:03 2017

@author: Bharadwaj
"""

import pandas as pd

df = pd.read_csv('All_data.csv')
train_data = []
test_data = []

df2 = pd.read_csv('movie-label.csv')
movieids = list(df2.loc[:, "movieid"])
label_dict = {key:0 for key in movieids}
for index, row in df2.iterrows():
     label_dict[row['movieid']] = row['label']
     
for index, row in df.iterrows():
    if row['movieid'] in label_dict.keys():
        train_data.append([row['movieid'], row['year'],
                           row['genres'],row['timestamp'],row['rating'],
                           row['genre_tfidf'], row['year_norm'], 
                           row['actor_tfidf'], row['moviename'], label_dict[row['movieid']]])
    else:
        test_data.append([row['movieid'],row['year'],
                           row['genres'],row['timestamp'],row['rating'],
                           row['genre_tfidf'], row['year_norm'], 
                           row['actor_tfidf'], row['moviename']]) 

df_train_data = pd.DataFrame(data=train_data, index=None, columns=['movieid','year',
'genres','timestamp','rating', 'genre_tfidf', 'year_norm', 'actor_tfidf','moviename' ,'label'])
    
df_test_data = pd.DataFrame(data=test_data, index=None, columns=['movieid','year',
'genres','timestamp','rating', 'genre_tfidf', 'year_norm', 'actor_tfidf', 'moviename'])
    
df_train_data.to_csv('Train_data.csv',index=False)
df_test_data.to_csv('Test_data.csv',index=False)