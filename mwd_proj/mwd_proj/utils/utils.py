from datetime import datetime
import django
import traceback
import csv
import os
os.environ['DJANGO_SETTINGS_MODULE'] = "mwd_proj.settings"
django.setup()
from django.db.models import Max, Min
from mwd_proj.phase1.models import *

def normalize_tables():
	max_val = long(MlRatings.objects.all().aggregate(Max('timestamp'))['timestamp__max'])
	min_val = long(MlRatings.objects.all().aggregate(Min('timestamp'))['timestamp__min'])
	#print max_val, min_val
	range_ = max_val - min_val
	print "MLRatings max {} min {} range {}".format(max_val, min_val, range_)
	records = MlRatings.objects.all()
	for record in records:
		val = long(record.timestamp)
		norm_val = float(val - min_val) / float(range_)
		record.norm_weight = str(norm_val)
		record.save()

        max_val = long(MlTags.objects.all().aggregate(Max('timestamp'))['timestamp__max'])
        min_val = long(MlTags.objects.all().aggregate(Min('timestamp'))['timestamp__min'])
        #print max_val, min_val
        range_ = max_val - min_val
	print "MLRatings max {} min {} range {}".format(max_val, min_val, range_)
        records = MlTags.objects.all()
        for record in records:
                val = long(record.timestamp)
                norm_val = float(val - min_val) / float(range_)
                record.norm_weight = str(norm_val)
                record.save()

if __name__ == "__main__":
	normalize_tables()