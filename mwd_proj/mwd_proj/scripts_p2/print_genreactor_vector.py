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

def main(genre, flag=0):
	"This model takes as input genre and model to give tag vector"
	try:
		tf()
		#initialize dict of all tags
		tf_dict = {}
		if flag == 0:
			all_tags_ = GenomeTags.objects.values_list('tag', flat=True)
		else:
			all_tags_ = GenomeTags.objects.values_list('tagid', flat=True)	
		for tag in all_tags_:
			tf_dict[tag] = 0.0

		total = Task2.objects.filter(genre__icontains=genre).aggregate(Sum('score'))['score__sum']
		records = Task2.objects.filter(genre__icontains=genre)
		max_val = float(Task2.objects.filter(genre__icontains=genre).aggregate(Max('score'))['score__max'])
		if max_val == float(0.0):
			max_val == 0.001
		if total == float(0.0):
			total == 0.001
		for record in records:
			if flag == 0:
				tf_dict[record.tag] = 0.5 + 0.5*(float(record.score / total) / max_val)
			else:
				_tag_id = (GenomeTags.objects.filter(tag=record.tag)[0]).tagid
				tf_dict[_tag_id] = 0.5 + 0.5*(float(record.score / total) / max_val)
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
			if flag == 0:
				tagobj = GenomeTags.objects.get(tag=tag)
			else:
				tagobj = GenomeTags.objects.get(tagid=tag)	
			count = MlTags.objects.filter(tagid=tagobj).aggregate(Sum('norm_weight'))['norm_weight__sum']
			#count = Task2.objects.filter(tag=tag).count()
			idf_score = math.log10(float(D)/float(count))
			idf_score = idf_score / max_
			tf_dict[tag] *= idf_score
		return tf_dict

	except Exception as e:
		traceback.print_exc()


def compute_tf_idf_actor_genre(genre):
	"Used by Kushal in 1b"

    cursor.execute("select actorid,actor_movie_rank from master where genre like %s", ('%' + genre + '%',))
    ret = cursor.fetchall()

    actors = [x[0] for x in ret]
    set_actors = set(actors)
    no_actors = len(actors)

    ranks = [x[1] for x in ret]
    sum_ranks = 0
    for r in ranks:
        sum_ranks+=rank_weights[r]

    cursor.execute("select count(DISTINCT genre) from master")
    no_genres = cursor.fetchall()[0][0]

    tfidf = []
    for a in actors:
        start = -1
        sum_weights = 0
        while True:
            try:
                i = actors.index(a, start + 1)
            except ValueError:
                break
            else:
                sum_weights += rank_weights[ranks[i]]
                start = i

        tf = sum_weights / sum_ranks
        cursor.execute("select count(distinct genre) from master where actorid=%s", (a,))
        genre_with_actor_a = cursor.fetchall()[0][0]
        idf = math.log(no_genres / genre_with_actor_a, 10)
        tf_idf = tf * idf;
        tfidf.append((a, tf_idf))
    return tfidf


def elapsedTime(starttime):
	elapsed = (time() - starttime)
	minu = int(elapsed) / 60
	sec = elapsed % 60
	print "\nElapsed time is min:",str(minu)," sec:",str(sec)


if __name__ == "__main__":
	tf()
	starttime = time()
	print main('Action', 1)
	elapsedTime(starttime)

	
	