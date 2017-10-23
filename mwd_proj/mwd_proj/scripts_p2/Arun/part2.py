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
from django.db.models import Sum, Avg, Q
import operator
import math
from django.db.models.functions import Lower
from mwd_proj.phase2.models import *
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector)
from collections import defaultdict
from mwd_proj.scripts_p2.part1 import compute_Semantics_1c
from mwd_proj.scripts_p2.Arun.part3 import coactor_matrix
from scipy.sparse.linalg import svds
import pandas as pd
from sklearn.cluster import KMeans


def compute_Semantics_2a(k=3, max_actors=3):
	"""Actor-actor similarity with svd and cluster grouping"""
	print "\n\n"
	actorobjs = ImdbActorInfo.objects.values_list('actorid','name')
	actor_dict = {x[0]:x[1] for x in actorobjs}
	#print actor_dict
	dict_semantics = defaultdict(dict)
	dict_grouping = defaultdict(list)
	matrix, actor_list = compute_Semantics_1c('SVD','Lillard, Matthew','cosine',10,5,False)

	u, sigma, Vt = svds(matrix,k)

	for row in u:
		max_ = max(row)
		min_ = min(row)
		for i in range(len(row)):
			row[i] = (row[i] - min_)/(max_ - min_)

	#for i in range(len(actor_list)):
	#	dict_grouping[group_list[i]].append(actor_list[i])

	i=1
	for row in u.T:
		id_ = np.argpartition(row, -max_actors)[-max_actors:]
		e1 = []
		for e in id_:
			local_actor_id = int(actor_list[e])
			new_dict = print_actor_vector.main(local_actor_id,0)
			sorted_dict = sorted(new_dict.items(), key=operator.itemgetter(1),reverse=True)
			for j in range(max_actors):
				if sorted_dict[j][1]!=0.0:
					if sorted_dict[j][0] in dict_semantics[i]:
						dict_semantics[i][sorted_dict[j][0]] += sorted_dict[j][1]
					else:
						dict_semantics[i][sorted_dict[j][0]] = sorted_dict[j][1]
		i+=1
	
	print("-------------------------------SEMANTICS-------------------------------")
	print("LATENT SEMANTIC 1")
	print sorted(dict_semantics[1].items(), key=operator.itemgetter(1),reverse=True)

	print("LATENT SEMANTIC 2")
	print sorted(dict_semantics[2].items(), key=operator.itemgetter(1),reverse=True)

	print("LATENT SEMANTIC 3")
	print sorted(dict_semantics[3].items(), key=operator.itemgetter(1),reverse=True)

	data = pd.DataFrame(u,columns=['Latent1','Latent2','Latent3'])
	kmeans = KMeans(n_clusters=3)
	kmeans.fit(data)
	labels = kmeans.predict(data)

	for i in range(len(actor_list)):
		dict_grouping[labels[i]].append(actor_dict[actor_list[i]])
	print("\n""-------------------------------GROUPING-------------------------------")
	print("GROUP 1")
	for i in dict_grouping[0]:
		print i,

	print("\nGROUP 2")
	for i in dict_grouping[1]:
		print i,

	print("\nGROUP 3")
	for i in dict_grouping[2]:
		print i,


def compute_Semantics_2b(k=3, max_actors=3):
	"""Coactor-Coactor similarity with svd and cluster grouping"""
	print "\n\n"
	actorobjs = ImdbActorInfo.objects.values_list('actorid','name')
	actor_dict = {x[0]:x[1] for x in actorobjs}
	dict_semantics = defaultdict(dict)
	dict_grouping = defaultdict(list)
	matrix, actor_dict_list = coactor_matrix()
	actor_list = []

	for key in actor_dict_list:
		actor_list.append(key)

	#perform svd on the matrix
	u, sigma, Vt = svds(matrix,k)

	#normalize each row of left singular matrix
	for row in u:
		max_ = max(row)
		min_ = min(row)
		for i in range(len(row)):
			row[i] = (row[i] - min_)/(max_ - min_)

	#for i in range(len(actor_list)):
	#	dict_grouping[group_list[i]].append(actor_list[i])

	#find list of index where the value for the semantic is the maximum
	i=1
	for row in u.T:
		id_ = np.argpartition(row, -max_actors)[-max_actors:]
		for e in id_:
			coactor_score= 0
			for j in range(len(actor_list)):
				coactor_score += matrix[e][j]
			dict_semantics[i][actor_dict[actor_list[e]]] = coactor_score
		i+=1
	
	print("-------------------------------SEMANTICS-------------------------------")
	print("LATENT SEMANTIC 1")
	for i in dict_semantics[1]:
		print i,

	print("\n""LATENT SEMANTIC 2")
	for i in dict_semantics[2]:
		print i,

	print("\n""LATENT SEMANTIC 3")
	for i in dict_semantics[3]:
		print i,

	#KMeans for finding clusters
	data = pd.DataFrame(u,columns=['Latent1','Latent2','Latent3'])
	kmeans = KMeans(n_clusters=3)
	kmeans.fit(data)
	labels = kmeans.predict(data)

	for i in range(len(actor_list)):
		dict_grouping[labels[i]].append(actor_dict[actor_list[i]])
	print("\n""-------------------------------GROUPING-------------------------------")
	print("GROUP 1")
	for i in dict_grouping[0]:
		print i,

	print("\nGROUP 2")
	for i in dict_grouping[1]:
		print i,

	print("\nGROUP 3")
	for i in dict_grouping[2]:
		print i,



def compute_Semantics_2d():
	"""Tensor decomposition on actor,movie,year and put actor into non-overlapping bins of latent semantics"""
	print "\n\n"
	tag_dict = {}
	taglist = Task7.objects.values_list('tagid', flat=True).distinct()
	tag_count = taglist.count()

	for n, each in enumerate(taglist):
		tag_dict[n] = each

	rating_dict = {}
	rate = Task7.objects.values_list('rating', flat=True).distinct()
	rating_count = rate.count()
	for n, each in enumerate(rate):
		rating_dict[n] = each

	movie_dict = {}
	mov = Task7.objects.values_list('movieid', flat=True).distinct()
	movie_count = mov.count()
	for n, each in enumerate(mov):
		movie_dict[n] = each

	tagobjs = GenomeTags.objects.values_list('tagid','tag')
	tag_mapping = {x[0]:x[1] for x in tagobjs}
	#tags = list(tagobjs)
	
	movieobjs = MlMovies.objects.values_list('movieid','moviename')
	movie_mapping = {x[0]:x[1] for x in movieobjs}
	

	# print(tag_count)
	# print(rating_count)
	# print(movie_count)

	# print tag_dict
	# print rating_dict
	# print movie_dict	

	# with open('tag_space_matrix/actor_dict.csv', 'wb') as csv_file:
	#     writer = csv.writer(csv_file)
	#     for key, value in sorted(actor_dict.items(),key=operator.itemgetter(1)):
	#        writer.writerow([value, key])
	# with open('tag_space_matrix/year_dict.csv', 'wb') as csv_file:
	#     writer = csv.writer(csv_file)
	#     for key, value in sorted(year_dict.items(),key=operator.itemgetter(1)):
	#        writer.writerow([value, key])
	# with open('tag_space_matrix/movie_dict.csv', 'wb') as csv_file:
	#     writer = csv.writer(csv_file)
	#     for key, value in sorted(movie_dict.items(),key=operator.itemgetter(1)):
	#        writer.writerow([value, key])
	tags = Task7.objects.values_list('tagid','movieid','rating')
	results = [[[0]*rating_count for i in range(movie_count)] for i in range(tag_count)]
	#print(len(results))
	#print(len(results[0]))
	#print(len(results[0][0]))
	
	#break
	inv_t = {v: k for k, v in tag_dict.iteritems()}
	inv_m = {v: k for k, v in movie_dict.iteritems()}
	inv_r = {v: k for k, v in rating_dict.iteritems()}
	
	for row in tags:
		#print(inv_t[row[0]])
		#row1 = MlRatings.objects.filter(userid=row1.userid.userid)
		results[inv_t[row[0]]][inv_m[row[1]]][inv_r[row[2]]]=1.0
		
	tensor = T.tensor(np.array(results))
	#print(tensor)
	factors = tensorly.decomposition.parafac(tensor,5)

	#ACTOR SEMANTICS
	#print(factors)
	#tucker
	#factors[0]=factors[1]
	#factors[1]=factors[2]
	#factors[2]=factors[3]
	#print("AFTER")
	#col_sums = factors[0].asnumpy().sum(axis=0)
	x=factors[0]
	#factors[0] = (x.asnumpy() - x.asnumpy().min(0)) / x.asnumpy().ptp(0)
	factors[0] = (x.asnumpy() - x.asnumpy().min(0)) / (x.asnumpy().max(0) - x.asnumpy().min(0))
	#print(factors[0])
	ls_1 = []
	ls_2 = []
	ls_3 = []
	ls_4 = []
	ls_5 = []
	# with open('tag_space_matrix/actor_dict.csv', mode='r') as infile:
	# 	reader = csv.reader(infile)
	# 	actor_dict = {rows[0]:rows[1] for rows in reader}


	for i in range(len(factors[0])):
	 row = factors[0][i]
	 #print(row)
	 num = np.ndarray.argmax(row)
	 val = max(row)/sum(row)
	 if num==0:
	   ls_1.append([tag_mapping[tag_dict[i]],val])
	 if num==1:
	   ls_2.append([tag_mapping[tag_dict[i]],val])
	 if num==2:
	   ls_3.append([tag_mapping[tag_dict[i]],val])
	 if num==3:
	   ls_4.append([tag_mapping[tag_dict[i]],val])
	 if num==4:
	   ls_5.append([tag_mapping[tag_dict[i]],val])
	  # for row in query:
	  #  ls_5.append([row['name'],val])
	print("\nTag Bins")
	print("LATENT SEMANTIC 1:")
	for i in reversed(sorted(ls_1,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 2:")
	for i in reversed(sorted(ls_2,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 3:")
	for i in reversed(sorted(ls_3,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 4:")
	for i in reversed(sorted(ls_4,key=lambda x: x[1])):
	 print(i)


	print("LATENT SEMANTIC 5:")
	for i in reversed(sorted(ls_5,key=lambda x: x[1])):
	 print(i)


	# MOVIE SEMANTICS
	x=factors[1]
	#factors[1] = (x.asnumpy() - x.asnumpy().min(0)) / x.asnumpy().ptp(0)
	factors[1] = (x.asnumpy() - x.asnumpy().min(0)) / (x.asnumpy().max(0) - x.asnumpy().min(0))
	#print(factors[1])
	ls_1 = []
	ls_2 = []
	ls_3 = []
	ls_4 = []
	ls_5 = []
	# with open('tag_space_matrix/movie_dict.csv', mode='r') as infile:
	# 	reader = csv.reader(infile)
	# 	actor_dict = {rows[0]:rows[1] for rows in reader}
	for i in range(len(factors[1])):
	 row = factors[1][i]
	 #print(row)
	 num = np.ndarray.argmax(row)
	 val = max(row)/sum(row)
	 if num==0:
	   ls_1.append([movie_mapping[movie_dict[i]],val])
	 if num==1:
	   ls_2.append([movie_mapping[movie_dict[i]],val])
	 if num==2:
	   ls_3.append([movie_mapping[movie_dict[i]],val])
	 if num==3:
	   ls_4.append([movie_mapping[movie_dict[i]],val])
	 if num==4:
	   ls_5.append([movie_mapping[movie_dict[i]],val])


	print("\nMovie Bins")
	print("LATENT SEMANTIC 1")
	for i in reversed(sorted(ls_1,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 2")
	for i in reversed(sorted(ls_2,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 3")
	for i in reversed(sorted(ls_3,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 4")
	for i in reversed(sorted(ls_4,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 5")
	for i in reversed(sorted(ls_5,key=lambda x: x[1])):
	 print(i)

	# YEAR SEMANTICS
	x=factors[2]
	factors[2] = (x.asnumpy() - x.asnumpy().min(0)) / (x.asnumpy().max(0) - x.asnumpy().min(0))
	ls_1 = []
	ls_2 = []
	ls_3 = []
	ls_4 = []
	ls_5 = []
	#print(len(factors[2]))
	for i in range(len(factors[2])):
	 row = factors[2][i]
	 #print(row)
	 num = np.ndarray.argmax(row)
	 val = max(row)/sum(row)
	 if num==0:
	  ls_1.append([rating_dict[i],val])
	 if num==1:
	  ls_2.append([rating_dict[i],val])
	 if num==2:
	  ls_3.append([rating_dict[i],val])
	 if num==3:
	  ls_4.append([rating_dict[i],val])
	 if num==4:
	  ls_5.append([rating_dict[i],val])

	print("\nRating Bins")
	print("LATENT SEMANTIC 1")
	for i in reversed(sorted(ls_1,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 2")
	for i in reversed(sorted(ls_2,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 3")
	for i in reversed(sorted(ls_3,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 4")
	for i in reversed(sorted(ls_4,key=lambda x: x[1])):
	 print(i)


	print("LATENT SEMANTIC 5")
	for i in reversed(sorted(ls_5,key=lambda x: x[1])):
	 print(i)

def compute_Semantics_2c():
	"""Tensor decomposition on actor,movie,year and put actor into non-overlapping bins of latent semantics"""
	print "\n\n"
	actor_dict = {}
	act = MovieActor.objects.values_list('actorid', flat=True).distinct()
	actor_count = act.count()

	for n, each in enumerate(act):
		actor_dict[n] = each

	year_dict = {}
	yr = MlMovies.objects.values_list('year', flat=True).distinct()
	year_count = yr.count()
	for n, each in enumerate(yr):
		year_dict[n] = each

	movie_dict = {}
	mov = MlMovies.objects.values_list('movieid', flat=True).distinct()
	movie_count = mov.count()
	for n, each in enumerate(mov):
		movie_dict[n] = each

	actorobjs = ImdbActorInfo.objects.values_list('actorid','name')
	actor_mapping = {x[0]:x[1] for x in actorobjs}
	
	movieobjs = MlMovies.objects.values_list('movieid','moviename')
	movie_mapping = {x[0]:x[1] for x in movieobjs}
	

	# print(actor_count)
	# print(year_count)
	# print(movie_count)

	# print actor_dict
	# print year_dict
	# print movie_dict	

	# with open('tag_space_matrix/actor_dict.csv', 'wb') as csv_file:
	#     writer = csv.writer(csv_file)
	#     for key, value in sorted(actor_dict.items(),key=operator.itemgetter(1)):
	#        writer.writerow([value, key])
	# with open('tag_space_matrix/year_dict.csv', 'wb') as csv_file:
	#     writer = csv.writer(csv_file)
	#     for key, value in sorted(year_dict.items(),key=operator.itemgetter(1)):
	#        writer.writerow([value, key])
	# with open('tag_space_matrix/movie_dict.csv', 'wb') as csv_file:
	#     writer = csv.writer(csv_file)
	#     for key, value in sorted(movie_dict.items(),key=operator.itemgetter(1)):
	#        writer.writerow([value, key])

	results = [[[0]*movie_count for i in range(year_count)] for i in range(actor_count)]
	# print(len(results))
	# print(len(results[0]))
	# print(len(results[0][0]))

	whole_table = MovieActor.objects.select_related('movieid').all()
	print("#################")
	# print(whole_table.count())
	inv_a = {v: k for k, v in actor_dict.iteritems()}
	inv_m = {v: k for k, v in movie_dict.iteritems()}
	inv_y = {v: k for k, v in year_dict.iteritems()}
	for row in whole_table:
		results[inv_a[row.actorid.actorid]][inv_y[row.movieid.year]][inv_m[row.movieid.movieid]]=1.0
		
	tensor = T.tensor(np.array(results))
	print(tensor)
	factors = tensorly.decomposition.parafac(tensor,5)

	#ACTOR SEMANTICS
	print(factors[0])
	print("AFTER")
	#col_sums = factors[0].asnumpy().sum(axis=0)
	x=factors[0]
	factors[0] = (x.asnumpy() - x.asnumpy().min(0)) / x.asnumpy().ptp(0)
	print(factors[0])
	ls_1 = []
	ls_2 = []
	ls_3 = []
	ls_4 = []
	ls_5 = []
	# with open('tag_space_matrix/actor_dict.csv', mode='r') as infile:
	# 	reader = csv.reader(infile)
	# 	actor_dict = {rows[0]:rows[1] for rows in reader}


	for i in range(len(factors[0])):
	 row = factors[0][i]
	 #print(row)
	 num = np.ndarray.argmax(row)
	 val = max(row)
	 if num==0:
	   ls_1.append([actor_mapping[actor_dict[i]],val])
	 if num==1:
	   ls_2.append([actor_mapping[actor_dict[i]],val])
	 if num==2:
	   ls_3.append([actor_mapping[actor_dict[i]],val])
	 if num==3:
	   ls_4.append([actor_mapping[actor_dict[i]],val])
	 if num==4:
	   ls_5.append([actor_mapping[actor_dict[i]],val])
	  # for row in query:
	  #  ls_5.append([row['name'],val])

	print("LATENT SEMANTIC 1")
	for i in reversed(sorted(ls_1,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 2")
	for i in reversed(sorted(ls_2,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 3")
	for i in reversed(sorted(ls_3,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 4")
	for i in reversed(sorted(ls_4,key=lambda x: x[1])):
	 print(i)


	print("LATENT SEMANTIC 5")
	for i in reversed(sorted(ls_5,key=lambda x: x[1])):
	 print(i)


	# MOVIE SEMANTICS
	x=factors[2]
	factors[2] = (x.asnumpy() - x.asnumpy().min(0)) / x.asnumpy().ptp(0)
	ls_1 = []
	ls_2 = []
	ls_3 = []
	ls_4 = []
	ls_5 = []
	# with open('tag_space_matrix/movie_dict.csv', mode='r') as infile:
	# 	reader = csv.reader(infile)
	# 	actor_dict = {rows[0]:rows[1] for rows in reader}
	for i in range(len(factors[2])):
	 row = factors[2][i]
	 #print(row)
	 num = np.ndarray.argmax(row)
	 val = max(row)
	 if num==0:
	   ls_1.append([movie_mapping[movie_dict[i]],val])
	 if num==1:
	   ls_2.append([movie_mapping[movie_dict[i]],val])
	 if num==2:
	   ls_3.append([movie_mapping[movie_dict[i]],val])
	 if num==3:
	   ls_4.append([movie_mapping[movie_dict[i]],val])
	 if num==4:
	   ls_5.append([movie_mapping[movie_dict[i]],val])



	print("LATENT SEMANTIC 1")
	for i in reversed(sorted(ls_1,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 2")
	for i in reversed(sorted(ls_2,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 3")
	for i in reversed(sorted(ls_3,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 4")
	for i in reversed(sorted(ls_4,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 5")
	for i in reversed(sorted(ls_5,key=lambda x: x[1])):
	 print(i)

	# YEAR SEMANTICS
	x=factors[1]
	factors[1] = (x.asnumpy() - x.asnumpy().min(0)) / x.asnumpy().ptp(0)
	ls_1 = []
	ls_2 = []
	ls_3 = []
	ls_4 = []
	ls_5 = []
	for i in range(len(factors[1])):
	 row = factors[1][i]
	 #print(row)
	 num = np.ndarray.argmax(row)
	 val = max(row)
	 if num==0:
	  ls_1.append([year_dict[i],val])
	 if num==1:
	  ls_2.append([year_dict[i],val])
	 if num==2:
	  ls_3.append([year_dict[i],val])
	 if num==3:
	  ls_4.append([year_dict[i],val])
	 if num==4:
	  ls_5.append([year_dict[i],val])


	print("LATENT SEMANTIC 1")
	for i in reversed(sorted(ls_1,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 2")
	for i in reversed(sorted(ls_2,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 3")
	for i in reversed(sorted(ls_3,key=lambda x: x[1])):
	 print(i)

	print("LATENT SEMANTIC 4")
	for i in reversed(sorted(ls_4,key=lambda x: x[1])):
	 print(i)


	print("LATENT SEMANTIC 5")
	for i in reversed(sorted(ls_5,key=lambda x: x[1])):
	 print(i)


def table_joiner():
	"""Creates a metadata table"""
	ar = MlRatings.objects.filter().values('movieid').annotate(score=Avg('rating'))
	avg_rating = {}
	for a in ar:
		avg_rating[a['movieid']] = a['score']

	Task7.objects.all().delete()
	tagobjs = MlTags.objects.all()
	for eachobj in tagobjs:
		tobjs = MlRatings.objects.filter(movieid=eachobj.movieid)
		for ea in tobjs:
			if ea.rating > avg_rating[eachobj.movieid.movieid]:
				Task7.objects.create(movieid=eachobj.movieid.movieid, tagid=eachobj.tagid.tagid, rating=ea.rating)


if __name__ == "__main__":
	a=compute_Semantics_2a(3,5)
	#print a
	b=compute_Semantics_2b(3,5) #Arguement 1 - Number of latent features, Arguement 2 - Number of actors a latent feature describes
	#print b
	#table_joiner()   #For prepopulation, run only once on new data
	g=compute_Semantics_2c()
	# print g
	h=compute_Semantics_2d()
	# print h
