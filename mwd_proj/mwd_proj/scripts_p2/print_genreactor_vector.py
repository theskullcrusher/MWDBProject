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

def tf():
	"This method prepopulates meta table for this phase2task1.b Task5 for faster processing of tf for Kushal"
	try:
		Task5.objects.all().delete()
		genres = MlMovies.objects.values_list('genres', flat=True)
		distinct_genres = []
		for genre in genres:
			distinct_genres.extend(genre.split(','))
		distinct_genres = [x.strip() for x in distinct_genres]
		distinct_genres = list(set(distinct_genres))

		tf_dict = {}
		for genre in distinct_genres:
			tf_dict[genre] = {}
			movies = MlMovies.objects.filter(genres__icontains=genre)
			for movie in movies:
				actors = MovieActor.objects.filter(movieid=movie)
				for actor in actors:
					norm_rank = 1.0 - actor.norm_rank
					score = float(norm_rank)
					if actor.actorid.actorid in tf_dict[genre].keys():
						tf_dict[genre][actor.actorid.actorid] += score
					else:
						tf_dict[genre][actor.actorid.actorid] = score
			keys = tf_dict[genre].keys()
			for key in keys:
				sc = tf_dict[genre][key]
				Task5.objects.create(genre=genre, actorid=key ,score=sc)
	except:
		traceback.print_exc()

def main(genre):
	"This model takes as input genre to give tfidf actor vector"
	try:
		#tf()
		#initialize dict of all actors
		tf_dict = {}
		all_actors_ = ImdbActorInfo.objects.values_list('actorid', flat=True)	
		for actorid in list(all_actors_):
			tf_dict[actorid] = 0.0

		total = Task5.objects.filter(genre__icontains=genre).aggregate(Sum('score'))['score__sum']
		records = Task5.objects.filter(genre__icontains=genre)
		max_val = float(Task5.objects.filter(genre__icontains=genre).aggregate(Max('score'))['score__max'])
		if max_val == float(0.0):
			max_val == 0.001
		if total == float(0.0):
			total == 0.001
		for record in records:
			tf_dict[record.actorid] = 0.5 + 0.5*(float(record.score / total) / max_val)

		#Calculate total D which is visualized as distinct no of (movie,genre) pairs
		genres = MlMovies.objects.values_list('genres', flat=True)
		distinct_genres = []
		for genre in genres:
			distinct_genres.extend(genre.split(','))
		distinct_genres = [x.strip() for x in distinct_genres]
		distinct_genres = list(set(distinct_genres))
		D = MlMovies.objects.filter(reduce(operator.or_, (Q(genres__icontains=x) for x in distinct_genres))).count()
		#normalize idf too
		max_ = math.log10(float(D))
		keys = tf_dict.keys()
		for actorid in keys:
			actorobj = ImdbActorInfo.objects.get(actorid=actorid)
			count = MovieActor.objects.filter(actorid=actorobj).aggregate(Sum('norm_rank'))['norm_rank__sum']
			idf_score = math.log10(float(D)/float(count))
			idf_score = idf_score / max_
			tf_dict[actorid] *= idf_score
		return tf_dict

	except Exception as e:
		traceback.print_exc()


def elapsedTime(starttime):
	elapsed = (time() - starttime)
	minu = int(elapsed) / 60
	sec = elapsed % 60
	print "\nElapsed time is min:",str(minu)," sec:",str(sec)


if __name__ == "__main__":
	tf()
	starttime = time()
	print main('Action')
	elapsedTime(starttime)

	
	