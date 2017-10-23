import sys, os
from datetime import datetime
import django
import traceback
os.environ['DJANGO_SETTINGS_MODULE']="mwd_proj.settings"
django.setup()
from mwd_proj.utils.utils2 import *
import traceback
from django.db.models import Sum
import operator
import math
from django.db.models.functions import Lower
from mwd_proj.phase2.models import *
from django.db.models import Q
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector, print_movie_vector)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
from math import log,exp
import pprint
from mysqlConn import DbConnect
import argparse
from numpy import *
import operator
import pandas as pd


def compute_Semantics_4(userid):

	#=====================================================================
	#Task:7 - Generate a rating of all movies for a user and sort them
	#=====================================================================

	#After reconstruction, we loose the column and row header names. so we need to do
	#some mapping.

	with open("R_final.csv") as f:
		ncols = len(f.readline().split('\t'))

	R_final = pd.DataFrame(loadtxt('R_final.csv',delimiter='\t', skiprows=1, usecols=range(1,ncols)))


	#Get user_ids and movie_ids

	with open("user_ids.csv") as f:
		ncols_u = len(f.readline().split('\t'))

	with open("movie_ids.csv") as f:
		ncols_m = len(f.readline().split('\t'))

	user_list = list(loadtxt('user_ids.csv',delimiter='\t', skiprows=1, usecols=range(1,ncols_u)))
	movie_list = list(loadtxt('movie_ids.csv',delimiter='\t', skiprows=1, usecols=range(1,ncols_m)))

	#print user_list
	#print movie_list
	user_index = user_list.index(float(userid))
	#print R_final.columns[user_index]

	movie_itr = 0
	movie_recommendations={}

	for index, row in R_final.iterrows():
		#print row[user_index]
		#Map the generated rating values for a user to a given movie. Get a ceiling rating.
		#print movie_list[movie_itr]
		movie_recommendations[movie_list[movie_itr]] = abs(ceil(row[user_index]))
		if movie_recommendations[movie_list[movie_itr]] <= 0:
			movie_recommendations[movie_list[movie_itr]] = 0.5
			# Such movies will be recommended below the 1 rating movie

		#print movie_recommendations[movie_list[movie_itr]]
		movie_itr+=1

	#print "--------Generated movie recommendations---------"
	#print movie_recommendations

	#============================================================
	#Get a list of user watched movies
	#============================================================

	userWatchedMovies = []
	usr = MlUsers.objects.get(userid=int(userid))
	result0 = MlRatings.objects.filter(userid=usr)
	for data in result0:
		#print type(data[0])
		userWatchedMovies.append(data.movieid.movieid)

	result0 = MlTags.objects.filter(userid=usr)
	for data in result0:
		userWatchedMovies.append(data.movieid.movieid)

	userWatchedMovies = list(set(userWatchedMovies))

	print "\n-----Watched movies-------"
	for watched_ids in userWatchedMovies:
		mv = MlMovies.objects.get(movieid=watched_ids)
		print "Movie details:\nName: {}; Genre {}".format(mv.moviename, mv.genres) 
	#print "----Watched movies-------"
	#print userWatchedMovies

	#=======================================================
	#Filter out ratings of watched movies
	#=======================================================
	userNotWatched={}

	for mv_id in movie_recommendations:
		mid = int(mv_id)
		#print type(mid)
		if mid in userWatchedMovies:
			#print "watched found"
			continue
		else:
			userNotWatched[mid] = movie_recommendations[mv_id]

	#print "---- movies not watched---"
	#print userNotWatched


	#print "----Sorted movie recommendations-----"
	movie_recommendations_sorted = sorted(userNotWatched.items(), key=operator.itemgetter(1), reverse=True)
	#print movie_recommendations_sorted
	#print movie_recommendations_sorted

	#Return top 5 unwatched movies in the generated recommendations
	print "\n-------Top 5 Recommended movies------"
	for i in range(0,5,1):
		#print movie_recommendations_sorted[i]
		mv = MlMovies.objects.get(movieid=int(movie_recommendations_sorted[i][0]))
		print "Movie details:\nName: {}; Genre {}".format(mv.moviename, mv.genres) 

if __name__ == "__main__":
	compute_Semantics_4(1027)