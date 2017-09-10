from mwdb_proj.phase1.models import *
from datetime import datetime
import django
from django.contrib.auth import User
import traceback
import csv
import os
os.environ['DJANGO_SETTINGS_MODULE'] = "mwdb_proj.settings"
django.setup()


def populate_db():
	"Populate the db"
	try:
		pass
		with open("../dataset_p1/genome-tags.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				GenomeTags.objects.create(tagid=row[0], tag=row[1])

		with open("../dataset_p1/imdb-actor-info.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				ImdbActorInfo.objects.create(actorid=row[0], name=row[1], gender=row[2].strip())

		with open("../dataset_p1/mlmovies.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				row[2] = row[2].replace("|", ",")
				MlMovies.objects.create(movieid=row[0], moviename=row[1], genres=row[2])

		with open("../dataset_p1/mlusers.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				MlUsers.objects.create(userid=row[0])

		with open("../dataset_p1/movie-actor.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				MovieActor.objects.create(movieid=row[0], actorid=row[1], actor_movie_rank=row[2])

		with open("../dataset_p1/mltags.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				dt = datetime.strptime(str(row[3]), "%m/%d/%Y  %I:%M:%S %p")
				row[3] = dt.strftime('%s')
				MlTags.objects.create(userid=row[0], movieid=row[1], tagid=row[2], timestamp=row[3])

		with open("../dataset_p1/mlratings.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				dt = datetime.strptime(str(row[4]), "%m/%d/%Y  %I:%M:%S %p")
				row[4] = dt.strftime('%s')
				MlRatings.objects.create(movieid=row[0], userid=row[1], imdbid=row[2], rating=row[3], timestamp=row[4])


	except Exception as e:
		traceback.print_exc()


if __name__ == "__main__":
	django.setup()
	populate_db()
