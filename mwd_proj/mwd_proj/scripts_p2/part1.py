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
import math
from django.db.models.functions import Lower
from mwd_proj.phase2.models import *
from django.db.models import Q
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector)

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


def compute_Semantics_1a(method, genre,k_topics):
	"""Here the data is (genre X tags) with each cell having Tf-IDF values for that genre and tag"""
	#All genres
	_genres = MlMovies.objects.values_list('genres', flat=True)
	genres = []
	for genre in _genres:
		genres.extend(genre.split(','))
	genres = [x.strip() for x in genres]
	genres = list(set(genres))
	#All tags
	tagobjs = GenomeTags.objects.values_list('tagid','tag')
	tags_dict = {x[0]:x[1] for x in tagobjs}
	tags = GenomeTags.objects.values_list('tagid', flat=True)

	'''Matrix Dataset'''
	V = sp.lil_matrix((len(genres), len(tags)))
	decomposed = []
	'''get tf-idfs vectors for genre-tag pairs and fill the matrix
		0 if genre-tag doesn't exist'''

	for i in range(len(genres)):
		# tf_idf = compute_tf_idf_movie(cur_movie,"TF-IDF")
		tf_idf = print_genre_vector.main(str(genres[i]), 1)
		for j in range(len(tags)):
			V[i, j] = tf_idf[tags[j]]

	if(method.upper() == 'SVD'):
		'''  SVD  Calculation '''
		U, sigma, Vt = svds(V, k=k_topics)
		sigma = np.diag(sigma)
		# print "\n\nSigma = \t",sigma
		print "\n\nU:", len(U), len(U[0]), "Sigma: ", sigma.shape, " V: ", Vt.shape, "\n\n"
		#print U
		decomposed = U
		print "For genre Latent semantics are:", U[genres.index(genre)]


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
		for i, gen in enumerate(genres):
			tobjects = Task2.objects.filter(genre=gen)
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

		lda = LDA(n_components=k_topics, max_iter=10000, learning_method="batch")
		lda.fit(V)
		Vt = lda.components_
		decomposed = lda.transform(V)
		lda = LDA(n_components=k_topics, max_iter=200, learning_method="batch")
		print "200 iterations: \n",lda.fit_transform(V)

	'''IN order to give Latenet Semantics some names: Normalize each column in feature factor matrix
					  and then pick top 5 tags somewhat describing that Latent Semantic '''
	#normalize columns for most discriminating feature finding
	normed_Vt = Vt/Vt.sum(axis=0)
	#print "\n\nHo ho!!\n", normed_Vt
	#print tags_dict
	for i in range(k_topics):
		idx = np.argpartition(-normed_Vt[i], 10)[:10]
		# print "What is this?", -np.partition(-normed_Vt[0], 5)[:5]
		print idx
		print "Latent Semantics: ", i + 1, " = "
		li = []
		for j in idx:
			li.append(tags_dict[tags[j]])
		print '\t', li, "\n"

	return decomposed


def compute_Semantics_1b(method, genre, k_topics):
	'''Here the data is (genre X actors) with each cell having Tf-IDF values for that genre and actor'''
	#All genres
	_genres = MlMovies.objects.values_list('genres', flat=True)
	genres = []
	for genre in _genres:
		genres.extend(genre.split(','))
	genres = [x.strip() for x in genres]
	genres = list(set(genres))

	#All actors
	actorobjs = ImdbActorInfo.objects.values_list('actorid','name')
	actors_dict = {x[0]:x[1] for x in actorobjs}
	actors = ImdbActorInfo.objects.values_list('actorid', flat=True)

	'''Matrix Dataset'''
	V = sp.lil_matrix((len(genres), len(actors)))
	decomposed = []
	'''get tf-idfs vectors for each genre w.r.t actors'''
	for i in range(len(genres)):
		tf_idf = print_genreactor_vector.main(str(genres[i]))
		for j in range(len(actors)):
			V[i, j] = tf_idf[actors[j]]

	if (method.upper() == 'SVD'):
		'''  SVD  Calculation '''
		U, sigma, Vt = svds(V, k=k_topics)
		sigma = np.diag(sigma)
		# print "\n\nSigma = \t",sigma
		print "\n\nU:", len(U), len(U[0]), "Sigma: ", sigma.shape, " V: ", Vt.shape, "\n\n"
		#print U
		#print "For genre Latent semantics are:", U[genres.index(genre)]
		decomposed = U

	if (method.upper() == 'PCA'):
		# standardizing data
		V = sp.csr_matrix(V).todense()
		V_std = StandardScaler().fit_transform(V)
		print "Stdandardized size: ", V_std.shape

		'''PCA::   Using Inbuilt library function'''
		sklearn_pca = PCA(n_components=k_topics)
		pca = sklearn_pca.fit(V_std)
		Vt = pca.components_
		# print Vt
		decomposed = pca.transform(V_std)

	if (method.upper() == 'LDA'):
		'''TO:DO://  Create matrix with doc as rows and words as column s with each cell having freq count not tf-idf'''
		for i, gen in enumerate(genres):
			tobjects = Task5.objects.filter(genre=gen)
			t_actors_id = tobjects.values_list('actorid', flat=True)
			for j in range(len(actors)):
				aid = actors[j]
				if aid in t_actors_id:
					V[i, j] = tobjects.get(actorid=int(aid)).score
				else:
					V[i, j] = 0.0

		lda = LDA(n_components=k_topics, max_iter=10000, learning_method="batch")
		lda.fit(V)
		Vt = lda.components_
		decomposed = lda.transform(V)
		lda = LDA(n_components=k_topics, max_iter=200, learning_method="batch")
		print "200 iterations: \n",lda.fit_transform(V)


	'''SVD,PCA :: IN order to give Latenet Semantics some names: Normalize each column in feature factor matrix
					  and then pick top 5 actors somewhat describing that Latent Semantic '''
	#normed_Vt = normalize(Vt, axis=0, norm='max')
	normed_Vt = Vt / Vt.sum(axis=0)

	for i in range(k_topics):
		idx = np.argpartition(-normed_Vt[i], 10)[:10]
		print "Latent Semantic: ", i + 1, " = "
		li = []
		for j in idx:
			li.append(actors_dict[actors[j]])
		print '\t', li, "\n"
	return decomposed


def compute_Semantics_1c(method, actor, measure, similarity_count=10, k_topics=5):
	'''Here the data is (actor X tag) with each cell having TF-IDF values for that Actor and Tag which we use to compute n nearest neighbors'''
	#All actors
	actorobjs = ImdbActorInfo.objects.values_list('actorid','name')
	actors_dict = {x[0]:x[1] for x in actorobjs}
	actors = list(ImdbActorInfo.objects.values_list('actorid', flat=True))
	#All tags
	tagobjs = GenomeTags.objects.values_list('tagid','tag')
	tags_dict = {x[0]:x[1] for x in tagobjs}
	tags = GenomeTags.objects.values_list('tagid', flat=True)

	'''Matrix Dataset'''
	V = sp.lil_matrix((len(actors), len(tags)))
	#V = np.zeros(shape=(len(actors), len(tags)))
	decomposed = []
	'''get tf-idfs vectors for each actor w.r.t tags'''
	
	for i in range(len(actors)):
		tf_idf = print_actor_vector.main(str(actors[i]), 1)
		for j in range(len(tags)):
			V[i, j] = tf_idf[tags[j]]

	ac_obj = ImdbActorInfo.objects.filter(name__icontains=actor).first()
	if ac_obj != None:
		pass
	else:
		print "Actor not found!!"
		exit()
	print "Top {} actors similar to {} are: ".format(similarity_count, actor)

	ac_index = actors.index(ac_obj.actorid)
	if (method.upper() == 'TF-IDF'):
		if measure.lower() == 'cosine':
			for i in range(len(actors)):
				for j in range(len(tags)):
					if V[i,j] == float(0.0):
						V[i,j] = 9999999.0
			decomposed = cosine_similarity(V)
			# orig_decomposed = decomposed
			# decomposed.sort(axis=1)
			# vect = decomposed[ac_index,-(similarity_count+1):]
			vect = decomposed[ac_index,:].argsort()[-(similarity_count+1):]

		elif measure.lower() == 'euclidean':
			decomposed = euclidean_distances(V)
			# orig_decomposed = decomposed
			# decomposed.sort(axis=1)
			# vect = decomposed[ac_index,:(similarity_count+1)]
			vect = decomposed[ac_index,:].argsort()[:(similarity_count+1)]
			
		vect = vect.tolist()
		orig_vect = decomposed[ac_index,:].tolist()
		actor_ids = []
		for each in vect:
			#o_index = orig_vect.index(each)
			#if o_index != ac_index:
			if each != ac_index:
				actor_ids.append(orig_vect[each])
		output = []
		for act in actors_dict:
			if act in actor_ids:
				output.append(actors_dict[act])
		print vect
		print orig_vect
		print actor_ids
		print output
		return decomposed


	if (method.upper() == 'SVD'):
		'''  SVD  Calculation '''
		U, sigma, Vt = svds(V, k=k_topics)
		sigma = np.diag(sigma)
		# print "\n\nSigma = \t",sigma
		print "\n\nU:", len(U), len(U[0]), "Sigma: ", sigma.shape, " V: ", Vt.shape, "\n\n"
		#print U
		#print "For genre Latent semantics are:", U[genres.index(genre)]
		decomposed = U

	normed_Vt = Vt / Vt.sum(axis=0)

	for i in range(k_topics):
		idx = np.argpartition(-normed_Vt[i], 10)[:10]
		print "Latent Semantic: ", i + 1, " = "
		li = []
		for j in idx:
			li.append(actors_dict[actors[j]])
		print '\t', li, "\n"
	return decomposed


if __name__ == "__main__":
	# a=compute_Semantics_1a('SVD','Action',4)
	# print a
	# b=compute_Semantics_1a('PCA','Action',4)
	# print b
	# c=compute_Semantics_1a('LDA','Action',4)
	# print c
	# d=compute_Semantics_1b('SVD','Action',4)
	# print d
	# e=compute_Semantics_1b('PCA','Action',4)
	# print e
	# f=compute_Semantics_1b('LDA','Action',4)
	# print f
	g=compute_Semantics_1c('TF-IDF','Lillard, Matthew','cosine',10,5)
	print g
	h=compute_Semantics_1c('TF-IDF','Lillard, Matthew','euclidean',10,5)
	print h
	pass