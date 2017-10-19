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
	"This method prepopulates meta table for this task name Task<number> for faster processing of tf"
	try:
		Task3.objects.all().delete()
		users = MlUsers.objects.all()
		tf_dict = {}
		for user in users:
			tf_dict[user.userid] = {}
			movies = MlTags.objects.filter(userid=user).values_list('movieid',flat=True)
			for movieid in movies:
				movie = MlMovies.objects.get(movieid=movieid)
				tags = MlTags.objects.filter(movieid=movie, userid=user)
				for tag in tags:
					norm_weight = tag.norm_weight
					score = float(norm_weight)
					if tag.tagid.tag in tf_dict[user.userid].keys():
						tf_dict[user.userid][tag.tagid.tag] += score
					else:
						tf_dict[user.userid][tag.tagid.tag] = score
			keys = tf_dict[user.userid].keys()
			for key in keys:
				sc = tf_dict[user.userid][key]
				Task3.objects.create(userid=user.userid, tag=key ,score=sc)
	except:
		traceback.print_exc()

def main(userid):
	"This model takes as input userid to give tfidf tag vector"
	try:
		#tf()
		tf_dict = {}
		all_tags_ = GenomeTags.objects.values_list('tag', flat=True)
		for tag in all_tags_:
			tf_dict[tag] = 0.0

		userid = int(userid)
		total = Task3.objects.filter(userid=userid).aggregate(Sum('score'))['score__sum']
		records = Task3.objects.filter(userid=userid)
		try:
			max_val = float(Task3.objects.filter(userid=userid).aggregate(Max('score'))['score__max'])
			if max_val == float(0.0):
				max_val == 0.001
			if total == float(0.0):
				total == 0.001
			#print "max ",max_val
		except:
			#No tag present for this user, return all empty dict
			return tf_dict
		for record in records:
			tf_dict[record.tag] = 0.5 + 0.5*(float(record.score / total) / max_val)

		#Calculate total D which is visualized as distinct no of (movie,user) pairs
		D = MlTags.objects.values('userid','movieid').distinct().count()
		#print D
		#normalize idf too
		max_ = math.log10(float(D))
		keys = tf_dict.keys()
		for tag in keys:
			tagobj = GenomeTags.objects.get(tag=tag)
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
	print main(146)
	elapsedTime(starttime)
