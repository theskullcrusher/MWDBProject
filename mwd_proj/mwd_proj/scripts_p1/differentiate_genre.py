from time import time
import sys, os
from datetime import datetime
import django
import traceback
os.environ['DJANGO_SETTINGS_MODULE']="mwd_proj.settings"
django.setup()
from mwd_proj.utils.utils1 import *
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
		Task4.objects.all().delete()
		genres = MlMovies.objects.values_list('genres', flat=True)
		distinct_genres = []
		for genre in genres:
			distinct_genres.extend(genre.split(','))
		distinct_genres = [x.strip() for x in distinct_genres]
		distinct_genres = list(set(distinct_genres))
		#print distinct_genres

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


def pdiff1(genre1, genre2, tags, tf_dict):
	"Calculates pdiff1 for the 2 genre provided"
	R = MlMovies.objects.filter(genres__icontains=genre1).values_list('movieid', flat=True).distinct().count()
	M = MlMovies.objects.filter(reduce(operator.or_, (Q(genres__icontains=x) for x in [genre1, genre2]))).values_list('movieid', flat=True).distinct().count()

	movies1 = MlMovies.objects.filter(genres__icontains=genre1)
	movies2 = MlMovies.objects.filter(reduce(operator.or_, (Q(genres__icontains=x) for x in [genre1, genre2])))
	for tag in tags:
		tagid = GenomeTags.objects.get(tag=tag)
		r1j = MlTags.objects.filter(Q(tagid=tagid) & reduce(operator.or_, (Q(movieid=x) for x in movies1))).count()
		m1j = MlTags.objects.filter(Q(tagid=tagid) & reduce(operator.or_, (Q(movieid=x) for x in movies2))).count()
		#print r1j, m1j, tagid.tagid, M, R
		try:
			if m1j - r1j != 0:
				val = (float(r1j*(M-R+r1j-m1j))/float((R-r1j)*(m1j-r1j)))
			else:
				val = 0.0
		except Exception as e:
			val = 0.0
			#print e
 		
 		#print val
		#val = abs(val)
		if val == 0.0:
			tf_dict[tag] = 0.0
		else:
			val1 = abs((float(r1j)/float(R)) - (float(m1j-r1j)/float(M-R)))
			#print val1
			tf_dict[tag] = math.log10(val) * val1
	
	sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1), reverse=True)
	print "Sorted P-DIFF1 tags:\n"
	for value in sorted_dict:
		#pass
		print value


def pdiff2(genre1, genre2, tags, tf_dict):
	"Calculates pdiff2 for the 2 genre provided"
	R = MlMovies.objects.filter(genres__icontains=genre2).count()
	M = MlMovies.objects.filter(reduce(operator.or_, (Q(genres__icontains=x) for x in [genre1, genre2]))).count()

	movies1 = MlMovies.objects.filter(genres__icontains=genre1)
	movies2 = MlMovies.objects.filter(genres__icontains=genre2)
	movies = movies1 | movies2
	for tag in tags:
		tagid = GenomeTags.objects.get(tag=tag)
		r1j = len(movies2) - MlTags.objects.filter(~Q(tagid=tagid) & reduce(operator.or_, (Q(movieid=x) for x in movies2))).count()
		m1j = len(movies) - MlTags.objects.filter(~Q(tagid=tagid) & reduce(operator.or_, (Q(movieid=x) for x in movies))).count()
		#print r1j, m1j, tagid.tagid
		try:
			if m1j - r1j != 0:
				val = (float(r1j*(M-R+r1j-m1j))/float((R-r1j)*(m1j-r1j)))
			else:
				val = 0.0
		except Exception as e:
			val = 0.0
			#print e

 		#print val
		#val = abs(val)
		if val == 0.0:
			tf_dict[tag] = 0.0
		else:
			val1 = abs((float(r1j)/float(R)) - (float(m1j-r1j)/float(M-R)))
			#print val1
			tf_dict[tag] = math.log10(val) * val1
	
	sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1), reverse=True)
	print "Sorted P-DIFF2 tags:\n"
	for value in sorted_dict:
		print value


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
		#Is A-B = B-A??. Current i've taken this condition to be true
		tf_dict = {}
		all_tags_ = GenomeTags.objects.values_list('tag', flat=True)
		for tag in all_tags_:
			tf_dict[tag] = 0.0		

		if model.lower().strip() == 'tf-idf-diff':
			#TF_IDF_DIFF model
			movies = MlMovies.objects.filter(reduce(operator.or_, (Q(genres__icontains=x) for x in [genre1, genre2])))
			D = movies.count()
			#D = Task4.objects.annotate(genre_lower=Lower('genre')).values_list('genre_lower', flat=True).distinct().count()
			#normalize idf too
			max_ = math.log10(float(D))

			#calculate tfidf score for tf_dict1
			movies1 = MlMovies.objects.filter(Q(genres__icontains=genre1))
			for tag in tf_dict1.keys():
				tagobj = GenomeTags.objects.get(tag=tag)
				count = MlTags.objects.filter(reduce(operator.or_, (Q(movieid=x) for x in movies1)), tagid=tagobj).aggregate(Sum('norm_weight'))['norm_weight__sum']
				#count = Task4.objects.filter(tag=tag).count()
				idf_score = math.log10(float(D)/float(count))
				idf_score = idf_score / max_
				tf_dict1[tag] *= idf_score

			#calculate tfidf score for tf_dict2
			movies2 = MlMovies.objects.filter(Q(genres__icontains=genre2))
			for tag in tf_dict2.keys():
				tagobj = GenomeTags.objects.get(tag=tag)
				count = MlTags.objects.filter(reduce(operator.or_, (Q(movieid=x) for x in movies2)), tagid=tagobj).aggregate(Sum('norm_weight'))['norm_weight__sum']
				#count = Task4.objects.filter(tag=tag).count()
				idf_score = math.log10(float(D)/float(count))
				idf_score = idf_score / max_
				tf_dict2[tag] *= idf_score

			#tf_dict final diff score
			for tag in tf_dict1.keys():
				if tag in tf_dict2.keys():
					tf_dict[tag] = tf_dict1[tag] - tf_dict2[tag]
					#print tag
				else:
					tf_dict[tag] = tf_dict1[tag]
			#for tags in tf_dict2 bt not in tf_dict1
			for tag in tf_dict2.keys():
				if tag not in tf_dict1.keys():
					tf_dict[tag] = tf_dict2[tag]

			sorted_dict = sorted(tf_dict.items(), key=operator.itemgetter(1), reverse=True)
			print "Sorted TF-IDF-DIFF tags:\n"
			for value in sorted_dict:
				print value

		elif model.lower().strip() == 'p-diff1':
			pdiff1(genre1, genre2, tf_dict1.keys(), tf_dict)
		elif model.lower().strip() == 'p-diff2':
			pdiff2(genre1, genre2, tf_dict1.keys(), tf_dict)

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
