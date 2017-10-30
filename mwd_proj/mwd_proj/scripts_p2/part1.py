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
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector, print_movie_vector)

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


def compute_Semantics_1a(method, genre_,k_topics):
	"""Here the data is (genre X tags) with each cell having Tf-IDF values for that genre and tag"""
	print "\n\n\n============================================"
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
		print "For genre",genre_,"Latent semantics are:", U[genres.index(genre_)]


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
		print "For genre",genre_,"Latent semantics are:", decomposed[genres.index(genre_)]

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

		lda = LDA(n_components=k_topics, max_iter=10000, learning_method="batch",evaluate_every=10,perp_tol=1e-12)
		lda.fit(V)
		Vt = lda.components_
		decomposed = lda.transform(V)
		print "For genre",genre_,"Latent semantics are:", decomposed[genres.index(genre_)]

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
		idx = np.argpartition(-normed_Vt[i], 10)[:10]
		# print "What is this?", -np.partition(-normed_Vt[0], 5)[:5]
		#rint idx
		print "Latent Semantics: ", i + 1, " = "
		li = []
		for j in idx:
			li.append(tags_dict[tags[j]])
		print '\t', li, "\n"

	return decomposed


def compute_Semantics_1b(method, genre_, k_topics):
	'''Here the data is (genre X actors) with each cell having Tf-IDF values for that genre and actor'''
	print "\n\n\n============================================"
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
		print "For genre",genre_,"Latent semantics are:", U[genres.index(genre_)]
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
		print "For genre",genre_,"Latent semantics are:", decomposed[genres.index(genre_)]

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

		lda = LDA(n_components=k_topics, max_iter=10000, learning_method="batch",evaluate_every=10,perp_tol=1e-12)
		lda.fit(V)
		Vt = lda.components_
		decomposed = lda.transform(V)
		print "For genre",genre_,"Latent semantics are:", decomposed[genres.index(genre_)]


	'''SVD,PCA :: IN order to give Latenet Semantics some names: Normalize each column in feature factor matrix
					  and then pick top 5 actors somewhat describing that Latent Semantic '''
	#normed_Vt = normalize(Vt, axis=0, norm='max')
	#normed_Vt = Vt / Vt.sum(axis=0)
	normed_Vt = Vt.copy()

	x = normed_Vt.max(axis=0)
	y = normed_Vt.min(axis=0)
	for i in range(len(normed_Vt)):
		for j in range(len(normed_Vt[0])):
			normed_Vt[i][j] = float(normed_Vt[i][j] - y[j]) / float(x[j] - y[j])

	for i in range(k_topics):
		idx = np.argpartition(-normed_Vt[i], 10)[:10]
		print "Latent Semantic: ", i + 1, " = "
		li = []
		for j in idx:
			li.append(actors_dict[actors[j]])
		print '\t', li, "\n"
	return decomposed


def compute_Semantics_1c(method, actor, measure, similarity_count=10, k_topics=5, p_flag=True):
	'''Here the data is (actor X tag) with each cell having TF-IDF values for that Actor and Tag which we use to compute n nearest neighbors'''
	print "\n\n\n============================================"
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
	if p_flag:
		print "\n*******Top {} actors similar to {} using {} and {} are: ".format(similarity_count, actor, method, measure)

	ac_index = actors.index(ac_obj.actorid)
	if (method.upper() == 'TF-IDF'):
		if measure.lower() == 'cosine':
			decomposed = cosine_similarity(V)

		elif measure.lower() == 'euclidean':
			decomposed = euclidean_distances(V)
			for i in range(len(actors)):
				for j in range(len(actors)):
					if decomposed[i,j] != float(0.0):
						decomposed[i,j] = 1.0/decomposed[i,j]

		vect = decomposed[ac_index,:].argsort()[-(similarity_count+1):]	
		vect = vect.tolist()
		orig_vect = decomposed[ac_index,:].tolist()
		actor_ids = []
		max_vals = []
		for each in vect[::-1]:
			#o_index = orig_vect.index(each)
			#if o_index != ac_index:
			if each != ac_index:
				actor_ids.append(actors[each])
				max_vals.append(orig_vect[each])
		output = []
		for act1 in actors_dict:
			if act1 in actor_ids:
				output.append(actors_dict[act1])
		if p_flag:
			print "\n",output
			print "\nMax vector values:",max_vals
			#print "Orig vector:",orig_vect
			print "ActorIds:",actor_ids
		return decomposed, actors

	if (method.upper() == 'SVD'):
		#calculate svd
		U, sigma, Vt = svds(V, k=k_topics)
		sigma = np.diag(sigma)
		# print "\n\nSigma = \t",sigma
		#print "\n\nU:", len(U), len(U[0]), "Sigma: ", sigma.shape, " V: ", Vt.shape, "\n\n"
		#print U
		#print "For genre Latent semantics are:", U[genres.index(genre)]
		#decomposed = U

		if measure.lower() == 'cosine':
			decomposed = cosine_similarity(U)

		elif measure.lower() == 'euclidean':
			decomposed = euclidean_distances(U)
			for i in range(len(actors)):
				for j in range(len(actors)):
					if decomposed[i,j] != float(0.0):
						decomposed[i,j] = 1.0/decomposed[i,j]

		vect = decomposed[ac_index,:].argsort()[-(similarity_count+1):]	
		vect = vect.tolist()
		orig_vect = decomposed[ac_index,:].tolist()
		actor_ids = []
		max_vals = []
		for each in vect[::-1]:
			#o_index = orig_vect.index(each)
			#if o_index != ac_index:
			if each != ac_index:
				actor_ids.append(actors[each])
				max_vals.append(orig_vect[each])
		output = []
		for act1 in actors_dict:
			if act1 in actor_ids:
				output.append(actors_dict[act1])
		if p_flag:
			print "\n",output
			print "\nMax vector values:",max_vals
			#print "Orig vector:",orig_vect
			print "ActorIds:",actor_ids
		return decomposed, actors


def compute_Semantics_1d(method, movie, similarity_count=10, k_topics=5, p_flag=True):
	'''Here the data is (actor X tag) with each cell having TF-IDF values for that Actor and Tag which we use to compute n nearest neighbors'''
	print "\n\n\n============================================"
	#All actors
	actorobjs = ImdbActorInfo.objects.values_list('actorid','name')
	actors_dict = {x[0]:x[1] for x in actorobjs}
	actors = list(ImdbActorInfo.objects.values_list('actorid', flat=True))
	actors_names = list(ImdbActorInfo.objects.values_list('name', flat=True))
	#All tags
	tagobjs = GenomeTags.objects.values_list('tagid','tag')
	tags_dict = {x[0]:x[1] for x in tagobjs}
	tags = GenomeTags.objects.values_list('tagid', flat=True)
	#All movies
	movieobjs = MlMovies.objects.values_list('movieid','moviename')
	movies_dict = {x[0]:x[1] for x in movieobjs}
	movies = list(MlMovies.objects.values_list('movieid', flat=True))
	movies_names = list(MlMovies.objects.values_list('moviename', flat=True))


	'''Matrix Dataset'''
	V = sp.lil_matrix((len(movies_names), len(tags)))
	V1 = sp.lil_matrix((len(actors_names), len(tags)))
	decomposed = []
	decomposed1 = []
	'''get tf-idfs vectors for each movie w.r.t tags'''
	
	for i in range(len(movies_names)):
		tf_idf = print_movie_vector.main(str(movies_names[i]), 1)
		for j in range(len(tags)):
			V[i, j] = tf_idf[tags[j]]

	for i in range(len(actors_names)):
		tf_idf = print_actor_vector.main(str(actors[i]), 1)
		for j in range(len(tags)):
			V1[i, j] = tf_idf[tags[j]]

	mo_obj = MlMovies.objects.filter(moviename__icontains=movie).first()
	if mo_obj != None:
		pass
	else:
		print "Movie not found!!"
		exit()
	if p_flag:
		print "\n*******Top {} actors similar to {} using {}  are: ".format(similarity_count, movie, method)

	ac_index = movies.index(mo_obj.movieid)
	if (method.upper() == 'TF-IDF'):
		V2 = np.dot(V, V1.T)
		# print V.shape
		# print V1.shape
		# print V2.shape
		# print V2[ac_index,:].shape
		vect = V2[ac_index,:].toarray().argsort()[0][-(similarity_count+10):]	
		vect = vect.tolist()
		orig_vect = (V2[ac_index,:].toarray().tolist())[0]
		#print orig_vect
		actor_ids = []
		max_vals = []
		for each in vect[::-1]:
			#print each
			#o_index = orig_vect.index(each)
			#if o_index != ac_index:
			#if each != ac_index:
			actor_ids.append(actors[each])
			max_vals.append(orig_vect[each])
		output = []
		print "Similar Actors who might have acted in the movie too:\n",actor_ids
		#Remove ids for actors in this movie
		act_remove = list(MovieActor.objects.filter(movieid=movies[ac_index]).values_list('actorid',flat=True))
		for ac in act_remove:
			try:
				i = actor_ids.index(ac)
				del max_vals[i]
				actor_ids.remove(ac)
			except:
				pass
		print "Actors who have acted in the movie:\n",act_remove
		print "Top actorids after removing the above actors:\n",actor_ids
		for act1 in actors_dict:
			if act1 in actor_ids:
				output.append(actors_dict[act1])
		if p_flag:
			print "\n",output[:10]
			print "\nMax vector values:",max_vals[:10]
			#print "Orig vector:",orig_vect
			print "ActorIds:",actor_ids[:10]
		return decomposed

	if (method.upper() == 'SVD'):
		#calculate svd
		U, sigma, Vt = svds(V, k=k_topics)
		U1, sigma1, Vt1 = svds(V1, k=k_topics)
		decomposed = np.dot(U,U1.T)
		# sigma = np.diag(sigma)
		# if measure.lower() == 'cosine':
		# 	decomposed = cosine_similarity(U)
		# elif measure.lower() == 'euclidean':
		# 	decomposed = euclidean_distances(U)
		# 	for i in range(len(movies)):
		# 		for j in range(len(tags)):
		# 			if decomposed[i,j] != float(0.0):
		# 				decomposed[i,j] = 1.0/decomposed[i,j]

		vect = decomposed[ac_index,:].argsort()[-(similarity_count+10):]	
		vect = vect.tolist()
		orig_vect = decomposed[ac_index,:].tolist()
		actor_ids = []
		max_vals = []
		for each in vect[::-1]:
			#o_index = orig_vect.index(each)
			#if o_index != ac_index:
			if each != ac_index:
				actor_ids.append(actors[each])
				max_vals.append(orig_vect[each])
		output = []
		#Remove ids for actors in this movie
		act_remove = list(MovieActor.objects.filter(movieid=movies[ac_index]).values_list('actorid',flat=True))
		if p_flag:
			print "Similar Actors who might have acted in the movie too:\n",actor_ids

		for ac in act_remove:
			try:
				i = actor_ids.index(ac)
				del max_vals[i]
				actor_ids.remove(ac)
			except:
				pass
		for act1 in actors_dict:
			if act1 in actor_ids:
				output.append(actors_dict[act1])
		if p_flag:
			print "Actors who have acted in the movie:\n",act_remove
			print "Top actorids after removing the above actors:\n",actor_ids
			print "\n",output[:10]
			print "\nMax vector values:",max_vals[:10]
			#print "Orig vector:",orig_vect
			print "ActorIds:",actor_ids[:10]
		return decomposed


if __name__ == "__main__":
	a=compute_Semantics_1a('SVD','Action',4)
	print a
	b=compute_Semantics_1a('PCA','Action',4)
	print b
	c=compute_Semantics_1a('LDA','Action',4)
	print c
	d=compute_Semantics_1b('SVD','Action',4)
	print d
	e=compute_Semantics_1b('PCA','Action',4)
	print e
	f=compute_Semantics_1b('LDA','Action',4)
	print f
	g,z=compute_Semantics_1c('TF-IDF','Lillard, Matthew','cosine',10,5,True)
	print g
	# h,z=compute_Semantics_1c('TF-IDF','Lillard, Matthew','euclidean',10,5,True)
	# print h
	i,z=compute_Semantics_1c('SVD','Lillard, Matthew','cosine',10,5,True)
	print i
	# j,z=compute_Semantics_1c('SVD','Lillard, Matthew','euclidean',10,5,True)
	# print j
	k=compute_Semantics_1d('TF-IDF','Swordfish',10,5,True)
	#print k
	m=compute_Semantics_1d('SVD','Swordfish',10,5,True)
	# #print m
	# n=compute_Semantics_1d('SVD','Harry Potter and the Prisoner of Azkaban',10,5,True)
	# #print n
	# o=compute_Semantics_1d('SVD','Pitch Black',10,5,True)
	# #print o

