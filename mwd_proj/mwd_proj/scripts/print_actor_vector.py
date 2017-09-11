import time
import sys, os
from mwd_proj.phase1.models import *
from datetime import datetime
import django
import traceback
os.environ['DJANGO_SETTINGS_MODULE'] = "mwd_proj.settings"
django.setup()
from mwd_proj.utils.utils import *
import traceback
from django.db.models import Sum
import operator
import math

def tf():
	try:
		Task1.objects.all().delete()
		actors = ImdbActorInfo.objects.all()
		tf_dict = {}
		for actor in actors:
			tf_dict[actor.actorid] = {}
			movies = MovieActor.objects.filter(actorid=actor)
			for movie in movies:
				norm_rank = 1.0 - movie.norm_rank
				tags = MlTags.objects.filter(movieid=movie.movieid)
				for tag in tags:
					norm_weight = tag.norm_weight
					score = float(norm_weight * norm_rank)
					if tag.tagid.tag in tf_dict[actor.actorid].keys():
						tf_dict[actor.actorid][tag.tagid.tag] += score
					else:
						tf_dict[actor.actorid][tag.tagid.tag] = score
			keys = tf_dict[actor.actorid].keys()
			for key in keys:
				sc = tf_dict[actor.actorid][key]
				Task1.objects.create(actorid=actor.actorid, tag=key ,score=sc)
	except:
		traceback.print_exc()

def main():
	try:
		actorid = int(sys.argv[1])
		model = str(sys.argv[2])
		total = Task1.objects.filter(actorid=actorid).aggregate(Sum('score'))['score__sum']
		records = Task1.objects.filter(actorid=actorid)
		tf_dict = {}
		actor = ImdbActorInfo.objects.get(actorid=actorid)
		print "Actor Information: ID-{}; Name-{}; Gender-{}".format(actor.actorid, actor.name, actor.gender)
		for record in records:
			tf_dict[record.tag] = float(record.score / total)
		if model.lower().strip() == 'tf':
		#print tf_dict
			sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1), reverse=True)
			print "Sorted TF tags:\n{}\n\n".format(sorted_dict)
		else:
		#print tf*idf dict
			D = Task1.objects.values('actorid').distinct().count()
			#normalize idf too
			max_ = math.log10(float(D))
			keys = tf_dict.keys()
			for tag in keys:
				count = Task1.objects.filter(tag=tag).count()
				idf_score = math.log10(float(D)/float(count))
				idf_score = idf_score / max_
				tf_dict[tag] *= idf_score
			sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1), reverse=True)
			print "Sorted TF-IDF tags:\n{}\n\n".format(sorted_dict)	
	except Exception as e:
		traceback.print_exc()


def elapsedTime(starttime):
	elapsed = (time() - starttime)
	minu = int(elapsed) / 60
	sec = elapsed % 60
	print "Elapsed time is min:",str(minu)," sec:",str(sec)


if __name__ == "__main__":
	#tf()
	main()
	
