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
from django.db.models import Q

def tf():
	"This method prepopulates meta table for this task name Task<number> for faster processing of tf"
	try:
		Task2.objects.all().delete()
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
				Task2.objects.create(genre=genre, tag=key ,score=sc)
	except:
		traceback.print_exc()

def main():
	"This model takes as input genre and model to give tag vector"
	try:
		#initialize dict of all tags
		tf_dict = {}
		all_tags_ = GenomeTags.objects.values_list('tag', flat=True)
		for tag in all_tags_:
			tf_dict[tag] = 0.0

		genre = str(sys.argv[1])
		model = str(sys.argv[2])
		total = Task2.objects.filter(genre__icontains=genre).aggregate(Sum('score'))['score__sum']
		records = Task2.objects.filter(genre__icontains=genre)
#		tf_dict = {}
		print "Genre Information: Name-{};".format(genre)
		for record in records:
			tf_dict[record.tag] = float(record.score / total)
			#print record.score, total
		if model.lower().strip() == 'tf':
			#print tf_dict
			sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1), reverse=True)
			print "Sorted TF tags:\n"
			for value in sorted_dict:
				print value
		else:
			#Calculate total D which is visualized as distinct no of (movie,genre) pairs
			genres = MlMovies.objects.values_list('genres', flat=True)
			distinct_genres = []
			for genre in genres:
				distinct_genres.extend(genre.split(','))
			distinct_genres = [x.strip() for x in distinct_genres]
			distinct_genres = list(set(distinct_genres))
			# D = 0
			# for genre in distinct_genres:
			D = MlMovies.objects.filter(reduce(operator.or_, (Q(genres__icontains=x) for x in distinct_genres))).count()
			#D = Task2.objects.annotate(genre_lower=Lower('genre')).values_list('genre_lower', flat=True).distinct().count()
			#print D
			#normalize idf too
			max_ = math.log10(float(D))
			keys = tf_dict.keys()
			for tag in keys:
				tagobj = GenomeTags.objects.get(tag=tag)
				count = MlTags.objects.filter(tagid=tagobj).aggregate(Sum('norm_weight'))['norm_weight__sum']
				#count = Task2.objects.filter(tag=tag).count()
				idf_score = math.log10(float(D)/float(count))
				idf_score = idf_score / max_
				tf_dict[tag] *= idf_score
			sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1), reverse=True)
			print "Sorted TF-IDF tags:\n"
			for value in sorted_dict:
				print value
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

	
	