import time
import sys, os
from mwd_proj.phase1.models import *
from datetime import datetime
import django
import traceback
os.environ['DJANGO_SETTINGS_MODULE'] = "mwdb_proj.settings"
django.setup()
from mwd_proj.utils.utils import *
import traceback

def main():
	try:
		actorid = str(sys.argv[1])
		model = str(sys.argv[2])
		








	except Exception as e:
		traceback.print_exc()





def elapsedTime(starttime):
	elapsed = (time() - starttime)
	minu = int(elapsed) / 60
	sec = elapsed % 60
	print "Elapsed time is min:",str(minu)," sec:",str(sec)


if __name__ == "__main__":
	main()
	
