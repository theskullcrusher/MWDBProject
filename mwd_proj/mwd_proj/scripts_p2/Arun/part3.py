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
import math
from django.db.models.functions import Lower
from mwd_proj.phase2.models import *
from django.db.models import Q
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector, part1)
from mwd_proj.scripts_p2.Arun import ppr

def compute_Semantics_3a(setActors):
	print "\n\n"
	#setActors = set([1860883,486691,1335137,901175])
	actor_dict = {}
	act = MovieActor.objects.values_list('actorid', flat=True).distinct()
	actor_count = act.count()
	for n, each in enumerate(list(act)):
		actor_dict[each] = n
	setIndex = set([])
	inv_a = {v: k for k, v in actor_dict.iteritems()}
	for actorid in setActors:
		#print(actor_dict[2312401])
		setIndex.add(actor_dict[actorid])
	results, xyz = part1.compute_Semantics_1c('TF-IDF','Lillard, Matthew','cosine',10,5,False)
	#print(len(results))
	#print((results[0]))
	nodes,s=ppr.closedform(setIndex,results,0.85)
	#print(s)
	#print(s)
	result = list(reversed(sorted(range(len(s)), key=lambda k: s[k])))
	till_which = len(setActors)+10
	print(result[:till_which])
	print("Seed Actors:")
	for actorid in setActors:
		print(ImdbActorInfo.objects.get(actorid=actorid).name)
	for ea in result[:till_which]:
		ac = ImdbActorInfo.objects.get(actorid=inv_a[ea])
		print(inv_a[ea], ac.name, s[ea])
		#print(inv_a[ea], ac.name, s[ea], print_actor_vector.main(inv_a[ea]))



def coactor_matrix():
	"""Gets coactor matrix"""
	print "\n\n"
	actor_dict = {}
	act = MovieActor.objects.values_list('actorid', flat=True).distinct()
	actor_count = act.count()

	for n, each in enumerate(list(act)):
		actor_dict[each] = n
	results = [[0]*actor_count for i in range(actor_count)]
	act = list(act)
	for i in range(len(act)):
	 ac = ImdbActorInfo.objects.get(actorid=act[i])
	 movies = MovieActor.objects.filter(actorid=ac)

	 for movie in movies:
	 	 #print movie.movieid.movieid
		 result1 = MovieActor.objects.filter(movieid=movie.movieid)
		 for res in result1:
		  #print res.actorid.actorid
		  #print(res.actorid.actorid)
		  results[i][actor_dict[res.actorid.actorid]]+=1.0 
	 	 
	for i in range(len(results)):
	 results[i][i] =0.0
	return results, actor_dict	


def compute_Semantics_3b(setActors):
	#setActors = set([2312401])
	results, actor_dict = coactor_matrix()
	setIndex = set([])
	for actorid in setActors:
		#print(actor_dict[2312401])
		setIndex.add(actor_dict[actorid])
	#print(setIndex)
	# with open("coactor_matrix.csv", "wb") as f:
	#    writer = csv.writer(f)
	#    writer.writerows(results)
	inv_a = {v: k for k, v in actor_dict.iteritems()}
#	nodes,s=ppr.closedform(setActors,results)
	nodes,s=ppr.closedform(setIndex,results,0.85)
	#print(s)
	#print(s)
	result = list(reversed(sorted(range(len(s)), key=lambda k: s[k])))
	#print(result[:10])
	till_which = len(setActors)+10
	print("Seed Actors:")
	for actorid in setActors:
		print(ImdbActorInfo.objects.get(actorid=actorid).name)
	for ea in result[:till_which]:
		ac = ImdbActorInfo.objects.get(actorid=inv_a[ea])
		print(inv_a[ea], ac.name, s[ea])

if __name__ == "__main__":
	setActors = set([1860883,486691,1335137,901175])
	compute_Semantics_3a(setActors)
	compute_Semantics_3b(setActors)
	