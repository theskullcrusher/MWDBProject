from time import time
import sys, os
from datetime import datetime
import django
import traceback
os.environ['DJANGO_SETTINGS_MODULE']="mwd_proj.settings"
django.setup()
from mwd_proj.utils.utils3 import *
import traceback
from django.db.models import Sum
import operator
import math
from django.db.models.functions import Lower
from mwd_proj.phase3.models import *
from django.db.models import Q
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from copy import deepcopy
from mwd_proj.scripts_p3 import print_movie_vector
import json
import time
from mwd_proj.scripts_p3.lshash import LSHash
from collections import OrderedDict


def compute_Semantics_3a(method, k_topics):
	"""Here the data is (movie X tags) with each cell having Tf-IDF values for that movie and tag"""
	print "\n\n\n============================================"
	movies = []
	tags = []
	values = {}
	s_time = time.time()

	with open('pre_movie_tfidf.json','r') as f:
		saved_dict = json.load(f)
		movies = saved_dict['movies']
		tags = saved_dict['tags']
		values = saved_dict['values']



	#All movies
	# movies = MlMovies.objects.values_list('moviename', flat=True)
	# movies = [x.strip() for x in movies]
	# movies = list(set(movies))
	#All tags
	tagobjs = GenomeTags.objects.values_list('tagid','tag')
	tags_dict = {x[0]:x[1] for x in tagobjs}
	# tags = list(GenomeTags.objects.values_list('tagid', flat=True))

	'''Matrix Dataset'''
	V = sp.lil_matrix((len(movies), len(tags)))
	decomposed = []
	'''get tf-idfs vectors for movie-tag pairs and fill the matrix
		0 if movie-tag doesn't exist'''

	for i in range(len(movies)):
		for j in range(len(tags)):
			V[i, j] = values[str(i)+','+str(j)]

	# to_save_dict = {}
	# to_save_dict['movies'] = movies 
	# to_save_dict['tags'] = tags
	# values = {}
	# #Run only for the first time
	# print len(movies)
	# print len(tags)
	# t = time.time()

	#one-time-thing
	# for i in range(len(movies)):
	# 	# tf_idf = compute_tf_idf_movie(cur_movie,"TF-IDF")
	# 	tf_idf = print_movie_vector.main(str(movies[i]), 1)
	# 	for j in range(len(tags)):
	# 		V[i, j] = tf_idf[tags[j]]
	# 		values[str(i)+','+str(j)] = tf_idf[tags[j]]
	# 	if i%500 == 0:
	# 		print i
	# 		t1 = time.time()
	# 		print "Min:",int(t1 - t)/60, " Sec:",(t1 - t)%60 
	# to_save_dict['values'] = values
	# with open('pre_movie_tfidf.json','w+') as f:
	# 	json.dump(to_save_dict, f)


	if(method.upper() == 'SVD'):
		'''  SVD  Calculation '''
		U, sigma, Vt = svds(V, k=k_topics)
		sigma = np.diag(sigma)
		# print "\n\nSigma = \t",sigma
		# print "\n\nU:", len(U), len(U[0]), "Sigma: ", sigma.shape, " V: ", Vt.shape, "\n\n"
		#print U
		decomposed = U
		

	if(method.upper() == 'PCA'):
		# standardizing data
		V = sp.csr_matrix(V).todense()
		V_std = StandardScaler().fit_transform(V)
		#print "Stdandardized size: ", V_std.shape

		'''PCA::   Using Inbuilt library function'''
		sklearn_pca = PCA(n_components=k_topics)
		pca = sklearn_pca.fit(V_std)
		Vt = pca.components_
		#print Vt
		decomposed = pca.transform(V_std)
		
	if (method.upper() == 'LDA'):
		'''TO:DO://  Create matrix with doc as rows and words as column s with each cell having freq count not tf-idf'''
		for i, gen in enumerate(movies):
			tobjects = Task2.objects.filter(movie=gen)
			t_tags = tobjects.values_list('tag', flat=True)
			tags_dict1 = deepcopy(tags_dict)
			inv_tags = {v: k for k, v in tags_dict1.iteritems()}
			t_tags_id = [inv_tags[x] for x in inv_tags if x in t_tags]
			for j in range(len(tags)):
				tid = tags[j]
				if tid in t_tags_id:
					V[i, j] = tobjects.get(tag=str(tags_dict[tid])).score
				else:
					V[i, j] = 0.0

		lda = LDA(n_components=k_topics, max_iter=10000, learning_method="batch",evaluate_every=10,perp_tol=1e-12)
		lda.fit(V)
		Vt = lda.components_
		decomposed = lda.transform(V)
		
	'''IN order to give Latenet Semantics some names: Normalize each column in feature factor matrix
					  and then pick top 5 tags somewhat describing that Latent Semantic '''
	#normalize columns for most discriminating feature finding
	#normed_Vt = Vt/Vt.sum(axis=0)
	normed_Vt = Vt.copy()

	x = normed_Vt.max(axis=0)
	y = normed_Vt.min(axis=0)
	for i in range(len(normed_Vt)):
		for j in range(len(normed_Vt[0])):
			normed_Vt[i][j] = float(normed_Vt[i][j] - y[j]) / float(x[j] - y[j])

	#print "\n\nHo ho!!\n", normed_Vt
	#print tags_dict
	for i in range(k_topics):
#		idx = np.argpartition(-normed_Vt[i], 10)[:10]
		idx = np.argpartition(-normed_Vt[i], 10)[:10]

		# print "What is this?", -np.partition(-normed_Vt[0], 5)[:5]
		#rint idx
		print "\nLatent Semantics: ", i + 1, " = "
		li = []
		we = []
		for j in idx:
			li.append(tags_dict[tags[j]])
			we.append(normed_Vt[i][j])
		tot = ""
		for n, each in enumerate(li):
			tot+=str(each)+":"+str(we[n])+","
		print tot[:-1] + '\n'

	e_time = time.time() - s_time
	print "\n\nElapsed Min:",int(e_time)/60, " Sec:",(e_time)%60
	return decomposed, movies


def compute_Semantics_3b(movie_list, layers=5, hash_size=10, input_dim=500):
	""" Creates an inmemory datastructure with layers and hash_size with given input movies"""
	lsh = LSHash(hash_size, input_dim, num_hashtables=layers)
	for movie_vector in movie_list:
		lsh.index(movie_vector)
	return lsh

def compute_Semantics_3c(lsh, query_point, measure="euclidean", r=None):
	return lsh.query(query_point, num_results=r, distance_func=measure)


if __name__ == "__main__":
	vectors, movies=compute_Semantics_3a('SVD',500)
	vectors = [list(x) for x in vectors]

	movie_list = vectors
	#movie_list.extend([list(vectors[3]), list(vectors[10]), list(vectors[25]), list(vectors[30])])
	#print "Input Movies to index are:", movies[3],', ', movies[10],', ', movies[25],', ',movies[30]

	lsh = compute_Semantics_3b(movie_list, layers=30, hash_size=10, input_dim=500)
	
	#input_movie = "Friday Night Lights"
	input_movie = "Harry Potter and the Prisoner of Azkaban"
	for n, mov in enumerate(movies):
		if mov == input_movie:
			break
	if n == len(movies):
		print "\nEntered movie not found"
		exit()

	print 'Movie number in list:',n
	query_point = vectors[n]
	print "\nMovie to query: ", movies[n]
	candidates_length, result = compute_Semantics_3c(lsh, query_point, measure="euclidean", r=None)
	#measure =  ("hamming", "euclidean", "true_euclidean", "centred_euclidean", "cosine", "l1norm")
	#print result
	result_length = len(result)
	r = 50
	if len(result) < r:
		r = len(result)

	output = []
	print "\nMost similar movies to the above movie are:"
	for i in xrange(r):
		x = movie_list.index(list(result[i][0]))
		output.append((movies[x],result[i][1]))

	print "\nSimilar movies:",output
	print "\nTotal considered movies:",candidates_length
	print "\nUnique considered movies:",result_length