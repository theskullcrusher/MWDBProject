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

#Argument parser
'''
parser = argparse.ArgumentParser()
parser.add_argument("USER")
args = parser.parse_args()
'''
def get_user_mvrating_DF():
	#===========================================================================
	#Generate user - movie_rating matrix.
	#For each movie, get its rating given by a user. If no rating then give zero
	#===========================================================================

	dd_users_mvrating = {}
	dd_av_rating_for_genre = {}
	dd_total_movie_for_genre = {}

	#Limit is for checking that algorithm works.
	results = MlUsers.objects.all()[:50]
	for usr in results:
		print "for user" , usr.userid
		dd_users_mvrating[usr.userid] = {}
		dd_av_rating_for_genre[usr.userid] = {}
		dd_total_movie_for_genre[usr.userid] = {}

		#Get all movies watched(and hence rated) by each user.
		result1 = MlRatings.objects.filter(userid=usr)
#        cur2.execute("SELECT movieid, rating FROM `mlratings` where userid = %s",usr)
		for data1 in result1:

			user_movie_id = data1.movieid
			user_movie_rating = data1.rating

			if user_movie_id in dd_users_mvrating[usr.userid]:
				continue
			else:
				print user_movie_id,user_movie_rating
				dd_users_mvrating[usr.userid][user_movie_id] = user_movie_rating

			#mlmovies_clean maps one movie to a single genre.
			genres = MlMovies.objects.filter(movieid=user_movie_id).first()
			genres = genres.split(', ')
			for genre in genres:
				genre = genre.strip()

			##Dunno what exactly is mlmovies_clean
			cur2.execute("SELECT genres FROM `mlmovies_clean` where movieid = %s", {user_movie_id,})
			result_gen = cur2.fetchone()
			for data in result_gen:
				if data[0] in dd_av_rating_for_genre[usr.userid:
					dd_av_rating_for_genre[usr.userid[data[0]] += user_movie_rating
					dd_total_movie_for_genre[usr.userid[data[0]] += 1
				else:
					dd_av_rating_for_genre[usr.userid[data[0]] = user_movie_rating;
					dd_total_movie_for_genre[usr.userid][data[0]] = 1

		#WE need to do this again for mltags because it does not have a rating,
		# give rating = avg rating give to a particular genre to by a user.
		print "Getting mltags data........."

		# Get all movies tagged by each user. If movie is only tagged and not rated, then give rating of 2 (avg).
		result2 = MlTags.objects.filter(userid=usr)
		for data in result2:
			#print data1
			user_movie_id = data.movieid
			##Dunno what exactly is mlmovies_clean
			cur2.execute("SELECT genres FROM `mlmovies_clean` where movieid = %s", {user_movie_id, })
			mv_genre = cur2.fetchone()[0]

			if user_movie_id in dd_users_mvrating[usr.]:
				continue
			else:
				print user_movie_id
				dd_users_mvrating[usr.userid][user_movie_id] = dd_av_rating_for_genre[usr.userid[mv_genre]/dd_total_movie_for_genre[usr.userid][mv_genre]

		#Make rating of other movies to zero.
		genreNames = list(MlMovies.objects.values_list('movieid',flat=True).distinct())

		for keyval in genreNames:
			if keyval in dd_users_mvrating[usr.userid]:
				continue
			else:
				dd_users_mvrating[usr.userid][key] = 0

	#pprint.pprint(dd_users_genre)
	usr_mvrating_matrix = pd.DataFrame(dd_users_mvrating)
	return usr_mvrating_matrix

if __name__ == "__main__":
	usr_mvrating_matrix = get_user_mvrating_DF()
	# usr_genre_matrix = usr_genre_matrix.T
	# pprint.pprint(usr_genre_matrix)
	usr_mvrating_matrix.to_csv("factorization_1_user_mvrating.csv", sep='\t')











