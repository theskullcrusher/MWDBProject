# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:06:10 2017

@author: Bharadwaj
"""

import pandas as pd

label_data = []

num_of_movies = int(input("Enter numer of movies: "))

for i in range(num_of_movies):
    movieid = int(input("Enter movieid: "))
    label = int(input("Enter label for {0}: ".format(movieid)))
    label_data.append([movieid, label])


df_all = pd.DataFrame(data=label_data,index=None, columns=['movieid','label'])

df_all.to_csv('movie-label.csv',index=False)