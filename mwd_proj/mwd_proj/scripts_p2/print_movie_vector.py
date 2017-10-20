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
from mwd_proj.phase2.models import *

def tf():
	"This method prepopulates meta table for this task name Task6 for faster processing of tf"
	try:
		Task6.objects.all().delete()
		tf_dict = {}
		# actors = ImdbActorInfo.objects.all()
		movies = MlMovies.objects.all()
		for movie in movies:
			tf_dict[movie.movieid] = {}

		for movie in movies:
			tags = MlTags.objects.filter(movieid=movie.movieid)
			for tag in tags:
				score = float(tag.norm_weight)				
				if tag.tagid.tagid in tf_dict[movie.movieid].keys():
					tf_dict[movie.movieid][tag.tagid.tagid] += score
				else:
					tf_dict[movie.movieid][tag.tagid.tagid] = score
			keys = tf_dict[movie.movieid].keys()
			for key in keys:
				sc = tf_dict[movie.movieid][key]
				Task6.objects.create(movieid=movie.movieid, tagid=key, score=sc)
	except:
		traceback.print_exc()

def main(movie, flag=1):
	"This model takes as input movie to give tfidf tag vector"
	try:
		#tf()
		#initialize dict of all tags
		tf_dict = {}

		if flag == 0:
			all_tags_ = list(GenomeTags.objects.values_list('tag', flat=True))
		else:
			all_tags_ = list(GenomeTags.objects.values_list('tagid', flat=True))
		for tag in all_tags_:
			tf_dict[tag] = 0.0


		movie = str(movie)
		movie = MlMovies.objects.filter(moviename__icontains=movie).first()
		if movie == None:
			print "Movie not found in the database..."
			exit()

		total = Task6.objects.filter(movieid=movie.movieid).aggregate(Sum('score'))['score__sum']
		records = Task6.objects.filter(movieid=movie.movieid)
		try:
			max_val = float(Task6.objects.filter(movieid=movie.movieid).aggregate(Max('score'))['score__max'])
		except:
			return tf_dict
		if max_val == float(0.0):
			max_val == 0.001
		if total == float(0.0):
			total == 0.001
		for record in records:
			if flag == 1:
				tf_dict[record.tagid] = 0.5 + 0.5*(float(record.score / total) / max_val)
			else:
				_tag = (GenomeTags.objects.filter(tagid=record.tagid)[0]).tag
				tf_dict[_tag] = 0.5 + 0.5*(float(record.score / total) / max_val)
		
		D = MovieActor.objects.values('movieid').distinct().count()
		#print D
		#D = Task6.objects.values('actorid').distinct().count()
		#normalize idf too
		max_ = math.log10(float(D))
		keys = tf_dict.keys()
		for tagid in keys:
			if flag == 1:
				tagobj = GenomeTags.objects.get(tagid=tagid)
			else:
				tagobj = GenomeTags.objects.get(tag=tagid)
			count = MlTags.objects.filter(tagid=tagobj).aggregate(Sum('norm_weight'))['norm_weight__sum']
			idf_score = math.log10(float(D)/float(count))
			idf_score = idf_score / max_
			tf_dict[tag] *= idf_score

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
	print main("Harry Potter and the Prisoner of Azkaban",1)
	#print main("Pitch Black",1)
	elapsedTime(starttime)
	
