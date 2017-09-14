from time import time
import sys, os
from datetime import datetime
import django
import traceback
os.environ['DJANGO_SETTINGS_MODULE']="mwd_proj.settings"
django.setup()
from mwd_proj.utils.utils import *
import traceback
from django.db.models import Sum
import operator
import math
from django.db.models.functions import Lower
from mwd_proj.phase1.models import *


def tf():
	"This method prepopulates meta table for this task name Task<number> for faster processing of tf"
	try:
		Task4.objects.all().delete()
		genres = MlMovies.objects.values_list('genres', flat=True)
		distinct_genres = []
		for genre in genres:
			distinct_genres.extend(genre.split(','))
		distinct_genres = [x.strip() for x in distinct_genres]
		distinct_genres = list(set(distinct_genres))
		print distinct_genres

		tf_dict = {}
		for genre in distinct_genres:
			tf_dict[genre] = {}
			movies = MlMovies.objects.filter(genres__icontains=genre)
			for movie in movies:
				tags = MlTags.objects.filter(movieid=movie.movieid)
				for tag in tags:
					norm_weight = tag.norm_weight
					score = float(norm_weight)
					if tag.tagid.tag in tf_dict[genre].keys():
						tf_dict[genre][tag.tagid.tag] += score
					else:
						tf_dict[genre][tag.tagid.tag] = score
			keys = tf_dict[genre].keys()
			for key in keys:
				sc = tf_dict[genre][key]
				Task4.objects.create(genre=genre, tag=key ,score=sc)
	except:
		traceback.print_exc()

def main():
	"This method calculates the diff between 2 genres using one of the 3 input models"
	try:
		genre1 = str(sys.argv[1])
		genre2 = str(sys.argv[2])
		model = str(sys.argv[3])
		
		#Calculations for genre1
		total1 = Task4.objects.filter(genre__icontains=genre1).aggregate(Sum('score'))['score__sum']
		records1 = Task4.objects.filter(genre__icontains=genre1)
		tf_dict1 = {}
		print "Genre1 Information: Name-{};".format(genre1)
		for record in records1:
			tf_dict1[record.tag] = float(record.score / total1)

		#Calculations for genre2
		total2 = Task4.objects.filter(genre__icontains=genre2).aggregate(Sum('score'))['score__sum']
		records2 = Task4.objects.filter(genre__icontains=genre2)
		tf_dict2 = {}
		print "Genre2 Information: Name-{};".format(genre2)
		for record in records2:
			tf_dict2[record.tag] = float(record.score / total2)

		#Calculations for genre diff
		total = total1 + total2
		all_tags = tf_dict1.keys()
		all_tags.extend(tf_dict2.keys())
		all_tags = list(set(all_tags))
		#initiatise combined dict
		tf_dict = {}

		#Is A-B = B-A??. Current i've taken this condition to be true
		for tag in all_tags:
			tf_dict[tag] = 0.0
		#tf_dict diff score
		for tag in tf_dict1.keys():
			if tag in tf_dict2:
				tf_dict[tag] = abs(tf_dict1[tag] - tf_dict2[tag])
			else:
				tf_dict[tag] = abs(tf_dict1[tag])

		#for tags in tf_dict2 bt not in tf_dict1
		for tag in tf_dict2.keys():
			if tag not in tf_dict1:
				tf_dict[tag] = abs(tf_dict2[tag])
		
		if model.lower().strip() == 'tf-idf-diff':
			D = Task4.objects.annotate(genre_lower=Lower('genre')).values_list('genre_lower', flat=True).distinct().count()

			#normalize idf too, here not being down because min will be different and not 0
			#max_ = math.log10(float(D))

			for tag in all_tags:
				count = Task4.objects.filter(tag=tag).count()
				idf_score = math.log10(float(D)/float(count))
			#	idf_score = idf_score / max_
				tf_dict[tag] *= idf_score
			sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1), reverse=True)
			print "Sorted TF-IDF-DIFF tags:\n"
			for value in sorted_dict:
				print value
		elif True:
			pass
		else:
			pass
	except Exception as e:
		traceback.print_exc()


def elapsedTime(starttime):
	elapsed = (time() - starttime)
	minu = int(elapsed) / 60
	sec = elapsed % 60
	print "\nElapsed time is min:",str(minu)," sec:",str(sec)


if __name__ == "__main__":
	#tf()
	starttime = time()
	main()
	elapsedTime(starttime)
