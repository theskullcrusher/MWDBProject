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
					if tag.tag in tf_dict[actor.actorid].keys():
						tf_dict[actor.actorid][tag.tag] += score
					else:
						tf_dict[actor.actorid][tag.tag] = score
			keys = tf_dict[actor.actorid].keys()
			for key in keys:
				sc = tf_dict[actor.actorid][key]
				Task1.objects.create(actorid=actor.actorid, tag=key ,score=sc )
	except:
		traceback.print_exc()

def idf():
	pass

def main():
	try:
		actorid = (sys.argv[1])
		model = str(sys.argv[2])
		if model.lower().strip() == 'tf':
			total = Task1.objects.filter(actorid=actorid).aggregate(Sum('score'))['score__sum']
			records = Task1.objects.filter(actorid=actorid)
			tf_dict = {}
			actor = ImdbActorInfo.objects.get(actorid=actorid)
			print "Actor Information: ID-{}, Name-{}, Gender-{}".format(actor.actorid, actor.name, actor.gender)
			for record in records:
				tf_dict[record.tag] = float(record.score / total)
			sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1)).reverse()
			print "\n Sorted tags:\n{}".format(sorted_dict)

		else:
			pass
	except Exception as e:
		traceback.print_exc()


def elapsedTime(starttime):
	elapsed = (time() - starttime)
	minu = int(elapsed) / 60
	sec = elapsed % 60
	print "Elapsed time is min:",str(minu)," sec:",str(sec)


if __name__ == "__main__":
	#main()
	tf()
	#idf()	
