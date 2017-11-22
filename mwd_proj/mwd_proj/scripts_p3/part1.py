import numpy as np
import operator
import scipy.io
import tensorly.backend as T
import tensorly.decomposition
import numpy as np
from sklearn.preprocessing import normalize
from time import time
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
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import math
from django.db.models.functions import Lower
from mwd_proj.phase2.models import *
from django.db.models import Q
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector,print_movie_vector, part1)
from mwd_proj.scripts_p2.Arun import ppr

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
		tf_idf = print_movie_vector.main(str(movies[i]), 1)
		for j in range(len(tags)):
			V[i, j] = tf_idf[tags[j]]

	decomposed = cosine_similarity(V)
	return decomposed, movie_dict

def compute_Semantics_1a():
	pass

def compute_Semantics_1b():
	pass

def compute_Semantics_1c():
	pass

def compute_Semantics_1d(setMovies):
	#setActors = set([2312401])
	results, movie_dict = movie_matrix()
	#print(movie_dict)
	pos_feedback = []
	neg_feedback = []
	breakFlag=True
	while(breakFlag):
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
		result = list(reversed(sorted(range(len(s)), key=lambda k: s[k])))
		#print(result[:10])
		till_which = len(setMovies)+5
		print("Seed Actors:")
		for movieid in setMovies:
			print(MlMovies.objects.get(movieid=movieid).moviename)
		for ea in result[len(setMovies):till_which]:
			mov = MlMovies.objects.get(movieid=inv_m[ea])
			print(inv_m[ea], mov.moviename,mov.genres, s[ea])
			print("Enter feedback: ")
			feed = int(raw_input())
			if feed == 1:
				pos_feedback.append(mov.movieid)
			elif feed == 0:
				neg_feedback.append(mov.movieid)
			else:
				breakFlag = False
				break
		setMovies = list(setMovies) + pos_feedback
		setMovies = list(set(setMovies))
		for sm in setMovies:
			index1 = movie_dict[sm]
			for nm in neg_feedback:
				index2 = movie_dict[nm]
				print(results[index1][index2])
				results[index1][index2] /= 2 
				results[index2][index1] /= 2
				print(results[index1][index2])


if __name__ == "__main__":
	setMovies = set([3906])
	#compute_Semantics_3a(setActors)
	compute_Semantics_1d(setMovies)
	