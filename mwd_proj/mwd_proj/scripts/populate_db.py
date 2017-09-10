from mwd_proj.phase1.models import *
from datetime import datetime
import django
import traceback
import csv
import os
os.environ['DJANGO_SETTINGS_MODULE'] = "mwdb_proj.settings"
django.setup()


def populate_db():
	"Populate the db"
	try:
		pass
		count = 0
		MovieActor.objects.all().delete()
		MlTags.objects.all().delete()
		MlRatings.objects.all().delete()
		GenomeTags.objects.all().delete()
		ImdbActorInfo.objects.all().delete()
		MlMovies.objects.all().delete()
		MlUsers.objects.all().delete()

		with open("../dataset_p1/genome-tags.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				try:
					GenomeTags.objects.create(tagid=row[0], tag=row[1])
				except:
					count += 1
					continue

		with open("../dataset_p1/imdb-actor-info.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				try:
					ImdbActorInfo.objects.create(actorid=row[0], name=row[1], gender=row[2].strip())
				except:
					count += 1
					continue

		with open("../dataset_p1/mlmovies.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				row[2] = row[2].replace("|", ",")
				try:
					MlMovies.objects.create(movieid=row[0], moviename=row[1], genres=row[2])
				except:
					count += 1
					continue


		with open("../dataset_p1/mlusers.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				try:
					MlUsers.objects.create(userid=row[0])
				except:
					count += 1
					continue

		with open("../dataset_p1/movie-actor.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
				try:
					MovieActor.objects.create(movieid=row[0], actorid=row[1], actor_movie_rank=row[2])
				except:
					count += 1
					continue

		with open("../dataset_p1/mltags.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
                                try:
					dt = datetime.strptime(str(row[3]), "%m/%d/%Y  %I:%M:%S %p")
					row[3] = int(dt.strftime('%s'))
                                except:
					dt = datetime.strptime(str(row[3]), "%Y-%m-%d %H:%M:%S")
                                        row[3] = int(dt.strftime('%s'))
				try:
					MlTags.objects.create(userid=row[0], movieid=row[1], tagid=row[2], timestamp=row[3])
				except:
					count += 1
					continue

		with open("../dataset_p1/mlratings.csv","rb") as f:
			rows = csv.reader(f)
			for n, row in enumerate(rows):
				if n == 0:
					continue
                                try:
                                        dt = datetime.strptime(str(row[4]), "%m/%d/%Y  %I:%M:%S %p")
                                        row[4] = int(dt.strftime('%s'))
                                except:
                                        dt = datetime.strptime(str(row[4]), "%Y-%m-%d %H:%M:%S")
                                        row[4] = int(dt.strftime('%s'))
				try:
					MlRatings.objects.create(movieid=row[0], userid=row[1], imdbid=row[2], rating=row[3], timestamp=row[4])
				except:
					count += 1
					continue
		print "Unsuccessful Insertions:", count


	except Exception as e:
		traceback.print_exc()


if __name__ == "__main__":
	django.setup()
	populate_db()
