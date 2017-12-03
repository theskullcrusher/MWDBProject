l# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:11:16 2017

@author: Bharadwaj
"""

import pandas as pd
from collections import defaultdict
from datetime import datetime
import time

start_time = time.time()


num_features = 4
colnames=['movieid', 'year', 'genres']

df_mlratings = pd.read_csv('mlratings.csv')
#print df_mlratings[']

#df_mlmovies = pd.read_csv('mlmovies.csv',index_col=0)
#print df_mlmovies['year']
all_ele=[]
for index,row in df_mlratings.iterrows():
    value = (datetime.strptime(row['timestamp'],'%Y-%m-%d %H:%M:%S'))
    all_ele.append(value)

min_time = min(all_ele)
for key in range(len(all_ele)):
    value = (all_ele[key] - min_time).total_seconds()
    all_ele[key] = value

df_mlratings['timestamp'] = all_ele
df_mlratings['rating'] = df_mlratings['rating'].astype(float)
df_mlratings['timestamp'] = df_mlratings['timestamp'].astype(float)

dict_avg_rating = defaultdict(list)
dict_avg_timestamp = defaultdict(list)
for index,row in df_mlratings.iterrows():
    dict_avg_rating[int(row['movieid'])].append(float(row['rating']))
    dict_avg_timestamp[int(row['movieid'])].append(float(row['timestamp']))

for key in dict_avg_rating:
    dict_avg_rating[key] = sum(dict_avg_rating[key])/float(len(dict_avg_rating[key]))
    dict_avg_timestamp[key] =sum(dict_avg_timestamp[key])/float(len(dict_avg_timestamp[key]))

df_mlmovies = pd.read_csv('mlmovies.csv')
number_of_movies = len(df_mlmovies['movieid'])
for i in range(len(colnames)-1):
    df_mlmovies[colnames[i]] = df_mlmovies[colnames[i]].astype(int)

all_data = [[0 for i in range(num_features+1)] for j in range(number_of_movies)]

i = 0
min_timestamp = min(dict_avg_timestamp)
max_timestamp = max(dict_avg_timestamp)
range_timestamp = max_timestamp - min_timestamp

for index,row in df_mlmovies.iterrows():
    temp_list = []
    for j in range(len(colnames)):
        temp_list.append(row[colnames[j]])
    temp_list.append(dict_avg_rating[row['movieid']])
    temp_list.append(float(dict_avg_timestamp[row['movieid']]-min_timestamp)/range_timestamp)
    all_data[i] = temp_list
    i+=1

df_all = pd.DataFrame(data=all_data,index=None, columns=['movieid','year',
'genres','rating','timestamp'])

df = pd.read_csv('mlmovies.csv')
genres_from_db = list(df.loc[:, "genres"])

distinct_genres = []
years = list(df.loc[:,"year"])
min_year = min(years)
max_year = max(years)
range_year = max_year - min_year

for g in genres_from_db:
    genres = g.split('|')
    for genre in genres:
        if genre not in distinct_genres:
            distinct_genres.append(genre)            

#idf
dict_idf = {key: 0 for key in distinct_genres}
total = 0
for index, row in df.iterrows():
    genres = str(row['genres']).split('|')
    for genre in genres:
        dict_idf[genre] += 1
    total += 1
    
    
for key in dict_idf.keys():    
    dict_idf[key] = dict_idf[key]/total

#tf
movies_from_db = list(df.loc[:, "movieid"])
dict_tfidf = {key: 0 for key in movies_from_db}
year_dict = {key: 0 for key in movies_from_db} 
movie_name_dict = {key: "" for key in movies_from_db}
for index, row in df.iterrows():
    genres = str(row['genres']).split('|')
    for genre in genres:
        value = (0.5)+(0.5)*float(1/len(genres))
        value = value*dict_idf[genre]
        dict_tfidf[row['movieid']] += value
    dict_tfidf[row['movieid']] = dict_tfidf[row['movieid']]/float(len(genres))
    year_dict[row['movieid']] = float(row['year']-min_year)/range_year
    movie_name_dict[row['movieid']] = row['moviename']

df_all['genre_tfidf'] = df_all['movieid'].map(dict_tfidf)
df_all['year_norm'] = df_all['movieid'].map(year_dict)
df_all['moviename'] = df_all['movieid'].map(movie_name_dict)

#############################actor
df2 = pd.read_csv('movie-actor.csv')
actors_from_db = list(df2.loc[:, "actorid"])
actor_ranks = list(df2.loc[:, "actor_movie_rank"])
distinct_actors = []

for actor in actors_from_db:
    if actor not in distinct_actors:
        distinct_actors.append(actor)
        
#idf
dict_idf = {key: 0 for key in distinct_actors}
total = len(actors_from_db)
min_actor_movie_rank = min(actor_ranks)
max_actor_movie_rank = max(actor_ranks)
range_of_actor_movie_ranks = max_actor_movie_rank - min_actor_movie_rank
for index, row in df2.iterrows():
    norm_actor_movie_rank = float(row['actor_movie_rank'] - min_actor_movie_rank)/float(range_of_actor_movie_ranks)
    dict_idf[row['actorid']] += float(1/total)*norm_actor_movie_rank*10000

#tf
movies_from_db = list(df2.loc[:, "movieid"])
distinct_movies = []
for movie in movies_from_db:
    if movie not in distinct_movies:
        distinct_movies.append(movie)

dict_tfidf = {key: 0 for key in distinct_movies}
dict_tfidf2 = {key: 0 for key in distinct_movies}

for index, row in df2.iterrows():
    dict_tfidf[row['movieid']] += 1

for index, row in df2.iterrows():
    dict_tfidf2[row['movieid']] += float(1/dict_tfidf[row['movieid']])*dict_idf[row['actorid']]

df_all['actor_tfidf'] = df_all['movieid'].map(dict_tfidf2)

df_all.to_csv('All_data.csv',index=False)
    

print("Done...")
print("--- %s seconds ---" % (time.time() - start_time))