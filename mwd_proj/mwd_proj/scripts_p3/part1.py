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
from django.db.models import Sum
import operator
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import math
from django.db.models.functions import Lower
from mwd_proj.phase2.models import *
from django.db.models import Q
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector, part1)
#from mwd_proj.scripts_p3 import print_movie_vector
from mwd_proj.scripts_p2.Arun import ppr

class NDSparseMatrix:
  def __init__(self):
    self.elements = {}

  def addValue(self, tuple, value):
    self.elements[tuple] = value

  def readValue(self, tuple):
    try:
      value = self.elements[tuple]
    except KeyError:
      # could also be 0.0 if using floats...
      value = 0.0
    return value

def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
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
		print(str(movies[i]))
		tf_idf = print_movie_vector.main(str(movies[i]), 1)
		for j in range(len(tags)):
			V[i, j] = tf_idf[tags[j]]

	decomposed = cosine_similarity(V)
	return decomposed, movie_dict

def compute_Semantics_1a():
	pass

def compute_Semantics_1b():
	lda = LDA(n_components=k_topics, max_iter=10000, learning_method="batch",evaluate_every=10,perp_tol=1e-12)
	lda.fit(V)
	Vt = lda.components_

def compute_Semantics_1c(userid):
	"""Tensor decomposition on tag,movie,user and put actor into non-overlapping bins of latent semantics"""
	print "\n\n"
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
			user_movie_list.append((movie_dict[movie],movie_score[movie]))
	result = reversed(sorted(user_movie_list,key=lambda x: x[1]))
	print("Watched Movies:")
	rows = MlRatings.objects.all().filter(userid=user_dict[index])
	for row in rows:
		print(row.movieid.movieid,row.movieid.moviename,row.movieid.genres)
	print("Recommended:")
	for a,b in result:
		mov = MlMovies.objects.get(movieid=a)
		print(a,mov.moviename,mov.genres,b)
		#print(a,b)

def compute_Semantics_1d(setMovies):
	#setActors = set([2312401])
	results, movie_dict = movie_matrix()
	with open("movie_sim.csv", "w") as f:
	    writer = csv.writer(f)
	    writer.writerows(results)
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
				#print(results[index1][index2])
				results[index1][index2] -= 0.01 
				results[index2][index1] -= 0.01
				#print(results[index1][index2])


if __name__ == "__main__":
	setMovies = set([3906])
	#compute_Semantics_3a(setActors)
	userid = 19379
	start_time = time.time()
	compute_Semantics_1c(userid)
	print("--- %s seconds ---" % (time.time() - start_time))
	#compute_Semantics_1d(setMovies)
	