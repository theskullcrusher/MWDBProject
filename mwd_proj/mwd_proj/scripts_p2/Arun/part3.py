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

def compute_Semantics_3a():
	setActors = set([74])
	results = part1.compute_Semantics_1c('TF-IDF','Lillard, Matthew','cosine',10,5,False)
	nodes,s=ppr.powerIteration(setActors,results)
	#print(s)
	print(s)
	result = list(reversed(sorted(range(len(s)), key=lambda k: s[k])))
	print(result[:10])


def compute_Semantics_3b():
	setActors = set([74])
	actor_dict = {}
	act = MovieActor.objects.values_list('actorid', flat=True).distinct()
	actor_count = act.count()
	for n, each in enumerate(list(act)):
		#print(n,each)
		#actor_list[n] = each
		actor_dict[each] = n

	#print actor_dict
	#print list(act)

	results = [[0]*actor_count for i in range(actor_count)]
	print(len(results[0]))
	print(len(results))
	act = list(act)
	for i in range(len(act)):
	 #print actor_list[i]
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
		
	# with open("coactor_matrix.csv", "wb") as f:
	#    writer = csv.writer(f)
	#    writer.writerows(results)

	nodes,s=ppr.closedform(setActors,results)
	#print(s)
	print(s)
	result = list(reversed(sorted(range(len(s)), key=lambda k: s[k])))
	print(result[:10])

if __name__ == "__main__":
	h=compute_Semantics_3b()
	print h
	pass
