import numpy as np
import operator
import scipy.io
import tensorly.backend as T
import tensorly.decomposition
import numpy as np
from sklearn.preprocessing import normalize
from time import time
import sys, os
import scipy.sparse as sp
from datetime import datetime
import csv
import time
import django
import traceback
os.environ['DJANGO_SETTINGS_MODULE']="mwd_proj.settings"
django.setup()
from mwd_proj.utils.utils2 import *
import traceback
import pandas as pd
from django.db.models import Sum
import operator
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import math
from django.db.models.functions import Lower
from mwd_proj.phase2.models import *
from django.db.models import Q
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector, print_movie_vector, part1)
#from mwd_proj.scripts_p3 import print_movie_vector
from mwd_proj.scripts_p2.Arun import ppr
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
from math import log,exp
import pprint
import argparse
from numpy import *
import operator
import pandas as pd


def getRelevance(mv_relevance):
	rel_mv = []
	irr_mv = []
	genre_rel_count={}
	genre_total_count={}
	p={}
	u={}
	mv_rel_sum = {}

	#Separate the relevant and irrelevant movies
	for mv in mv_relevance:
		if mv_relevance[mv] == 0:
			irr_mv.append(mv)
		else:
			rel_mv.append(mv)
    
    #Get count r for a genre i.e number of relevant movies that have a genre
	
	for mv in rel_mv:
		genres = []
		result_gen = MlMovies.objects.values_list('genres').filter(movieid=mv)
		for val in result_gen:
			genres.extend(val[0].split(","))
		for val in genres:
			if val in genre_rel_count:
				genre_rel_count[val]+=1
			else:
				genre_rel_count[val] = 1        
	#get count of all movies that have a genre
	for mv in mv_relevance:
		genres = []
		result_gen = MlMovies.objects.values_list('genres').filter(movieid=mv)
		for val in result_gen:
			genres.extend(val[0].split(","))
		for val in genres:
			if val in genre_total_count:
				genre_total_count[val]+=1
			else:
				genre_total_count[val] = 1     
	#print(genre_rel_count)
	#print(genre_total_count)
	#R = total number of relevant retrieved items
	R = float(len(rel_mv))
	
	#N = total number of items
	N = 5
	
	#Get relevant and irrevelant probabilities
	for gen in genre_rel_count:
		p[gen] = float((genre_rel_count[gen] + 0.5 ) / (R + 1))
		u[gen] = float((genre_total_count[gen] - genre_rel_count[gen] + 0.5) / (N - R + 1))
		#print "p = ",p[gen]," u = ",u[gen]

	mov = list(MlMovies.objects.values_list('movieid', flat=True))
	for mv_id in mov:
		genres = []
		result_gen = MlMovies.objects.values_list('genres').filter(movieid=mv_id)
		for val in result_gen:
			genres.extend(val[0].split(","))
		#print(mv_id,genres)
		mv_rel_sum[mv_id] = 0;
		for genKey in genres:
			gen = genKey
			if gen in genre_rel_count:
				mv_rel_sum[mv_id] += math.log(float(p[gen]*(1-u[gen])) / (u[gen]*(1-p[gen])))
			else:
				continue
    #change this
	return mv_rel_sum


def movie_matrix():
	'''Here the data is (actor X tag) with each cell having TF-IDF values for that Actor and Tag which we use to compute n nearest neighbors'''
	print "\n\n\n============================================"
	#All actors
	movieobjs = MlMovies.objects.values_list('movieid','moviename')
	#movies_dict = {x[0]:x[1] for x in movieobjs}
	movies = list(MlMovies.objects.values_list('moviename', flat=True))
	mov = list(MlMovies.objects.values_list('movieid', flat=True))
	#All tags
	movie_dict = {}
	for n, each in enumerate(list(mov)):
		movie_dict[each] = n
	tagobjs = GenomeTags.objects.values_list('tagid','tag')
	tags_dict = {x[0]:x[1] for x in tagobjs}
	tags = GenomeTags.objects.values_list('tagid', flat=True)

	'''Matrix Dataset'''
	V = sp.lil_matrix((len(movies), len(tags)))
	#V = np.zeros(shape=(len(actors), len(tags)))
	decomposed = []
	'''get tf-idfs vectors for each actor w.r.t tags'''
	
	for i in range(len(movies)):
		#print(str(movies[i]))
		tf_idf = print_movie_vector.main(str(movies[i]), 1)
		for j in range(len(tags)):
			V[i, j] = tf_idf[tags[j]]

	decomposed = cosine_similarity(V)
	return decomposed, movie_dict

def compute_Semantics_1a(userid):
	#Precomputation:
	#1. Run preprocessor_model3.py
	#2. Run preprocessor_model3_comp.py	
	
	#============================================
	#Task:1(a) - Movie recommendation using SVD
	#============================================
	
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
	return userNotWatched

def compute_Semantics_1b(userid):
	#Precomputation:
	#1. Run preprocessor_model3_lda.py
	#2. Run preprocessor_model3_comp_lda.py

	#=============================================
	#Task:1(b) - Movie Recommendation using LDA
	#=============================================

	#After reconstruction, we loose the column and row header names. so we need to do
	#some mapping.

	with open("R_final_lda.csv") as f:
		ncols = len(f.readline().split('\t'))

	R_final = pd.DataFrame(loadtxt('R_final_lda.csv',delimiter='\t', skiprows=1, usecols=range(1,ncols)))


	#Get user_ids and movie_ids

	with open("user_ids_lda.csv") as f:
		ncols_u = len(f.readline().split('\t'))

	with open("movie_ids_lda.csv") as f:
		ncols_m = len(f.readline().split('\t'))

	user_list = list(loadtxt('user_ids_lda.csv',delimiter='\t', skiprows=1, usecols=range(1,ncols_u)))
	movie_list = list(loadtxt('movie_ids_lda.csv',delimiter='\t', skiprows=1, usecols=range(1,ncols_m)))

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
	return userNotWatched

def compute_Semantics_1c(userid):
	"""Tensor decomposition on tag,movie,user and put actor into non-overlapping bins of latent semantics"""
	print "\n\n"
	setMovies = MlRatings.objects.values_list("movieid").filter(userid=userid)
	setMovies = list(set([mov[0] for mov in setMovies]))

	tag_dict = {}
	taglist = MlRatings.objects.values_list('rating', flat=True).distinct()
	tag_count = taglist.count()
	#tag_count = 6
	for n, each in enumerate(taglist):
		tag_dict[n] = each

	user_dict = {}
	#user = MlRatings.objects.values_list('userid', flat=True).distinct()[:6000]
	user = MlRatings.objects.values_list('userid', flat=True).distinct()
	user_count = user.count()
	for n, each in enumerate(user):
		user_dict[n] = each

	movie_dict = {}
	mov = MlRatings.objects.values_list('movieid', flat=True).distinct()
	movie_count = mov.count()
	for n, each in enumerate(mov):
		movie_dict[n] = each

	# tagobjs = GenomeTags.objects.values_list('tagid','tag')
	# tag_mapping = {x[0]:x[1] for x in tagobjs}
	# #tags = list(tagobjs)
	
	movieobjs = MlMovies.objects.values_list('movieid','moviename')
	movie_mapping = {x[0]:x[1] for x in movieobjs}
	print(tag_count)
	print(movie_count)
	print(user_count)
	results = [[[0]*tag_count for i in range(movie_count)] for i in range(user_count)]
	#whole_table = MlRatings.objects.all()[:2000]
	whole_table = MlRatings.objects.all()
	inv_u = {v: k for k, v in user_dict.iteritems()}
	inv_m = {v: k for k, v in movie_dict.iteritems()}
	inv_t = {v: k for k, v in tag_dict.iteritems()}
	index = inv_u[userid]
	#print("whole_table")
	#print(whole_table.count())
	counter = 0
	for row in whole_table:
		#print(counter)
		counter+=1
		# print(inv_u[row.userid.userid],inv_m[row.movieid.movieid],inv_t[row.rating])
		results[inv_u[row.userid.userid]][inv_m[row.movieid.movieid]][inv_t[row.rating]]=1.0

	tensor = T.tensor(np.array(results))
	factors = tensorly.decomposition.parafac(tensor,3)
	recons = tensorly.kruskal_to_tensor(factors)
	#recons = results
	#tested for (25,88)g , (30,123)b, (150,497)okish,
	#index = 25
	movie_score = {}
	user_movie_list = []
	print("user: "+ str(user_dict[index]))
	for movie in range(len(recons[index])):
			#print("movie: "+str(movie))
			for rating in range(len(recons[index][movie])):
				#print("rating: "+str(rating))
				if recons[index][movie][rating].asscalar()> 0.0:
					if movie in movie_score:
						movie_score[movie] += float(rating+1)*(float(recons[index][movie][rating].asscalar()))
					else:
						movie_score[movie] = float(rating+1)*(float(recons[index][movie][rating].asscalar()))
			if movie not in movie_score:
				movie_score[movie] = 0.0
			#print("Score:")
			#print(movie_score[movie])
			if movie_dict[movie] not in setMovies:
				user_movie_list.append((movie_dict[movie],movie_score[movie]))
			else:
				user_movie_list.append((movie_dict[movie],0.0))

	breakFlag=True
	while(breakFlag):
		user_movie_dict = {}
		for k,v in user_movie_list:
			#print(k,v)
			user_movie_dict[k] = v
		user_movie_list = list(user_movie_dict.items())
		result = list(reversed(sorted(user_movie_list,key=lambda x: x[1])))
		till_which = 5
		#result = [item for item in list3 if item not in list(setMovies)]
		print("Watched Movies:")
		rows = MlRatings.objects.all().filter(userid=user_dict[index])
		for row in rows:
			print(row.movieid.movieid,row.movieid.moviename,row.movieid.genres)
		print("Recommended:")
		for a,b in result[:till_which]:
			mov = MlMovies.objects.get(movieid=a)
			print(a,mov.moviename,mov.genres,b)
		feedback = {}
		
		for ea,s in result[:till_which]:
			print("Enter feedback for: "+str(ea)+"...Hit X to exit")
			feed = int(raw_input())
			if feed == 'X':
				breakFlag = False
			feedback[ea] = feed
		
		movie_vector = getRelevance(feedback)
		for k,v in movie_vector.items():
			if v==0.0:
				v=0.0001
			user_movie_dict[k] *= v
		user_movie_list = list(user_movie_dict.items())

def compute_Semantics_1d(userid):
	#setActors = set([2312401])
	results, movie_dict = movie_matrix()
	with open("movie_sim.csv", "w") as f:
	    writer = csv.writer(f)
	    writer.writerows(results)
	#print(movie_dict)
	setMovies = MlRatings.objects.values_list("movieid").filter(userid=userid)
	setMovies = list(set([mov[0] for mov in setMovies]))
	print(setMovies)
	
	setIndex = set([])
	for movieid in setMovies:
		#print(actor_dict[2312401])
		setIndex.add(movie_dict[movieid])
	#print(setIndex)
	# with open("coactor_matrix.csv", "wb") as f:
	#    writer = csv.writer(f)
	#    writer.writerows(results)
	inv_m = {v: k for k, v in movie_dict.iteritems()}
	#	nodes,s=ppr.closedform(setActors,results)
	nodes,s=ppr.closedform(setIndex,results,0.85)
	#print(s)
	#print(s)
	breakFlag=True
	while(breakFlag):
		list3 = list(reversed(sorted(range(len(s)), key=lambda k: s[k])))
		result = [item for item in list3 if inv_m[item] not in list(setMovies)]
		#print(result[:10])
		till_which = len(setMovies)+5
		print("Watched Movies:")
		for movieid in setMovies:
			print(MlMovies.objects.get(movieid=movieid).moviename,MlMovies.objects.get(movieid=movieid).genres)
		print("Recommended Movies:")
		for ea in result[len(setMovies):till_which]:
			mov = MlMovies.objects.get(movieid=inv_m[ea])
			print(inv_m[ea], mov.moviename,mov.genres, s[ea])
		feedback = {}
		for ea in result[len(setMovies):till_which]:
			print("Enter feedback for: "+str(inv_m[ea])+"...Hit X to exit")
			feed = int(raw_input())
			if feed == 'X':
				breakFlag = False
			feedback[inv_m[ea]] = feed
		
		movie_vector = getRelevance(feedback)
		for k,v in movie_vector.items():
			if v==0.0:
				v=0.0001
			s[movie_dict[k]] *= v
		
		#print(movie_vector)
		# setMovies = list(setMovies) + pos_feedback
		# setMovies = list(set(setMovies))
		# for sm in setMovies:
		# 	index1 = movie_dict[sm]
		# 	for nm in neg_feedback:
		# 		index2 = movie_dict[nm]
		# 		#print(results[index1][index2])
		# 		results[index1][index2] -= 0.01 
		# 		results[index2][index1] -= 0.01
		# 		#print(results[index1][index2])


def compute_Semantics_1e(movie_svd,movie_lda,movie_tensor,movie_pagerank):
	'''Get all the movie vectors from each of the method, normalize them and compute a single value'''	
	movie_svd_sorted = sorted(movie_svd.items(), key=operator.itemgetter(1))
	svd_max = movie_svd_sorted[0][1]	
	svd_min = movie_svd_sorted[len(movie_svd_sorted)-1][1]
	for mv,v in movie_svd.iteritems():
		v = float(v - svd_min)/float(svd_max - svd_min)

	movie_lda_sorted = sorted(movie_lda.items(), key=operator.itemgetter(1))
	lda_max = movie_lda_sorted[0][1]	
	lda_min = movie_lda_sorted[len(movie_svd_sorted)-1][1]
	for mv,v in movie_lda.iteritems():
		v = float(v - lda_min)/float(lda_max - lda_min)	


	movie_tensor_sorted = sorted(movie_tensor.items(), key=operator.itemgetter(1))
	tensor_max = movie_tensor_sorted[0][1]	
	tensor_min = movie_tensor_sorted[len(movie_svd_sorted)-1][1]
	for mv,v in movie_tensor.iteritems():
		v = float(v - tensor_min)/float(tensor_max - tensor_min)

	movie_pagerank_sorted = sorted(movie_pagerank.items(), key=operator.itemgetter(1))
	pgrank_max = movie_pagerank_sorted[0][1]	
	pgrank_min = movie_pagerank_sorted[len(movie_svd_sorted)-1][1]
	for mv,v in movie_pagerank.iteritems():
		v = float(v - pgrank_min)/float(pgrank_max - pgrank_min)

	movie_all={}

	for mv,v in movie_svd:
		lda_v = movie_lda[mv]
		tensor_v = movie_tensor[mv]
		pgrank_v = movie_pagerank[mv]
		movie_all[mv] = v*lda_v*tensor_v*pgrank_v
	
	return movie_all

def compute_Feedback(userNotWatched)
	'''Currently computes feedback of SVD and LDA'''	
	while(True):
		#print "----Sorted movie recommendations-----"
		movie_recommendations_sorted = sorted(userNotWatched.items(), key=operator.itemgetter(1), reverse=True)
		#print movie_recommendations_sorted
		
		feedback = {}
		#Return top 5 unwatched movies in the generated recommendations
		print "\n-------Top 5 Recommended movies------"
		for i in range(0,5,1):
			#print movie_recommendations_sorted[i]
			mv = MlMovies.objects.get(movieid=int(movie_recommendations_sorted[i][0]))
			print "Movie details:\nName: {}; Genre {}".format(mv.moviename, mv.genres) 
			feedback[movie_recommendations_sorted[i][0]] = 0;
		
		print "-------Submit your feedback (relevant :'1', irrelevant :'0')-------- : "
		k=1
	    	for key in feedback:
			print "Movie", k
			feedback[key] = raw_input("Feedback : ")
			k+=1


		#Get the probabilistic relevance feedback values for all movies (prf)
		prf_movie = getRelevance(feedback)

		#Update the Rating vals (if value is 0, set to 0.0001):
		for key in userNotWatched.items():
			if key in prf_movie.items():
			    userNotWatched[key] *= prf_movie[key] + 0.0001


def compute_Recommendation(method,userid):
	'''Main funcntion to run by passing appropriate method and userID'''
	if(method.upper() == 'SVD' or method.upper() == 'LDA'):
		movie_matrix= compute_Semantics_1a(userid)
		compute_Feedback(movie_matrix)

	#Create similar methods for 1c and 1d

	if(method.upper() == 'ALL'):
		movie_svd = compute_Semantics_1a(userid)
		movie_lda = compute_Semantics_1b(userid)
		movie_tensor = compute_Semantics_1c(userid)
		movie_pagerank = compute_Semantics_1d(userid)
		movie_all = compute_Semantics_1e(movie_svd,movie_lda,movie_tensor,movie_pagerank)
		compute_Feedback(movie_all)



if __name__ == "__main__":
	userid = 88
	#userid = 19379
	start_time = time.time()
	#compute_Semantics_1a(userid)
	compute_Semantics_1b(userid)
	#compute_Semantics_1c(userid)
	#compute_Semantics_1d(userid)
	#compute_Semantics_1e(userid)
	print("--- %s seconds ---" % (time.time() - start_time))
	
	
