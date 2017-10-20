import csv
import numpy as np
import MySQLdb
from sqlalchemy import create_engine
from db_config import *
import operator
import scipy.io
import mat4py as m4pt
import tensorly.backend as T
import tensorly.decomposition
import numpy as np

engine = create_engine("mysql://"+get_user()+":"+get_pswd()+"@localhost/"+get_db(), pool_recycle=3600)
connection  = engine.connect()
connection.execute("drop view if exists avg_movie_rating;")
connection.execute("CREATE VIEW avg_movie_rating AS SELECT movieid,AVG(rating) as avg_rating from mlratings group by movieid;")
connection.execute("drop view if exists tag_movie_rating;")
connection.execute("CREATE VIEW tag_movie_rating AS SELECT tagid,mltags.movieid,rating FROM mlratings JOIN mltags ON mlratings.movieId = mltags.movieId WHERE rating>(SELECT avg_rating from avg_movie_rating where avg_movie_rating.movieid=mltags.movieid);")
#connection.execute("CREATE VIEW tag_movie_rating AS SELECT tagid,mltags.movieid,rating FROM mlratings JOIN mltags ON mlratings.movieId = mltags.movieId WHERE rating>0;")
tag_count = 0
tag_dict = {}
query = connection.execute("select DISTINCT tagid from tag_movie_rating;")
for row in query:
 tag_dict[row['tagid']] = tag_count
 tag_count+=1

rating_count = 0
rating_dict = {}
query = connection.execute("select DISTINCT rating from tag_movie_rating;")
for row in query:
 rating_dict[row['rating']] = rating_count
 rating_count+=1

movie_dict={}
movie_count = 0
query = connection.execute("select DISTINCT movieid from tag_movie_rating;")
for row in query:
 movie_dict[row['movieid']] = movie_count
 movie_count+=1

print(tag_count)
print(rating_count)
print(movie_count)

with open('tag_space_matrix/tag_dict.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in sorted(tag_dict.items(),key=operator.itemgetter(1)):
       writer.writerow([value, key])
with open('tag_space_matrix/rating_dict.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in sorted(rating_dict.items(),key=operator.itemgetter(1)):
       writer.writerow([value, key])
with open('tag_space_matrix/movie_dict_2d.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in sorted(movie_dict.items(),key=operator.itemgetter(1)):
       writer.writerow([value, key])

results = [[[0]*rating_count for i in range(movie_count)] for i in range(tag_count)]
print(len(results))
print(len(results[0]))
print(len(results[0][0]))

whole_table = connection.execute("select DISTINCT(tagid),movieid,rating from tag_movie_rating;")

for row in whole_table:
  results[tag_dict[row['tagid']]][movie_dict[row['movieid']]][rating_dict[row['rating']]]=1
	
#with open("tag_space_matrix/actormovieyear_tensor.csv", "wb") as f:
   #writer = csv.writer(f)
   #writer.writerows(results)

tensor = T.tensor(np.array(results))
print(tensor)
factors = tensorly.decomposition.parafac(tensor,5)
print(factors)
#TAG SEMANTICS
col_sums = factors[0].asnumpy().sum(axis=0)
factors[0] = factors[0].asnumpy()/col_sums[np.newaxis,:]
ls_1 = []
ls_2 = []
ls_3 = []
ls_4 = []
ls_5 = []
with open('tag_space_matrix/tag_dict.csv', mode='r') as infile:
    reader = csv.reader(infile)
    actor_dict = {rows[0]:rows[1] for rows in reader}
for i in range(len(factors[0])):
 row = factors[0][i]
 #print(row)
 num = np.ndarray.argmax(row)
 if num==0:
  query = connection.execute("select tag from genome_tags where tagId="+actor_dict[str(i)]+";")
  for row in query:
   ls_1.append(row['tag'])
 if num==1:
  query = connection.execute("select tag from genome_tags where tagId="+actor_dict[str(i)]+";")
  for row in query:
   ls_2.append(row['tag'])
 if num==2:
  query = connection.execute("select tag from genome_tags where tagId="+actor_dict[str(i)]+";")
  for row in query:
   ls_3.append(row['tag'])
 if num==3:
  query = connection.execute("select tag from genome_tags where tagId="+actor_dict[str(i)]+";")
  for row in query:
   ls_4.append(row['tag'])
 if num==4:
  query = connection.execute("select tag from genome_tags where tagId="+actor_dict[str(i)]+";")
  for row in query:
   ls_5.append(row['tag'])

print("LATENT SEMANTIC 1")
print(ls_1)

print("LATENT SEMANTIC 2")
print(ls_2)

print("LATENT SEMANTIC 3")
print(ls_3)

print("LATENT SEMANTIC 4")
print(ls_4)

print("LATENT SEMANTIC 5")
print(ls_5)

# MOVIE SEMANTICS
col_sums = factors[1].asnumpy().sum(axis=0)
factors[1] = factors[1].asnumpy()/col_sums[np.newaxis,:]
ls_1 = []
ls_2 = []
ls_3 = []
ls_4 = []
ls_5 = []
with open('tag_space_matrix/movie_dict_2d.csv', mode='r') as infile:
    reader = csv.reader(infile)
    actor_dict = {rows[0]:rows[1] for rows in reader}
for i in range(len(factors[1])):
 row = factors[1][i]
 #print(row)
 num = np.ndarray.argmax(row)
 if num==0:
  query = connection.execute("select moviename from mlmovies where movieid="+actor_dict[str(i)]+";")
  for row in query:
   ls_1.append(row['moviename'])
 if num==1:
  query = connection.execute("select moviename from mlmovies where movieid="+actor_dict[str(i)]+";")
  for row in query:
   ls_2.append(row['moviename'])
 if num==2:
  query = connection.execute("select moviename from mlmovies where movieid="+actor_dict[str(i)]+";")
  for row in query:
   ls_3.append(row['moviename'])
 if num==3:
  query = connection.execute("select moviename from mlmovies where movieid="+actor_dict[str(i)]+";")
  for row in query:
   ls_4.append(row['moviename'])
 if num==4:
  query = connection.execute("select moviename from mlmovies where movieid="+actor_dict[str(i)]+";")
  for row in query:
   ls_5.append(row['moviename'])

print("LATENT SEMANTIC 1")
print(ls_1)

print("LATENT SEMANTIC 2")
print(ls_2)

print("LATENT SEMANTIC 3")
print(ls_3)

print("LATENT SEMANTIC 4")
print(ls_4)

print("LATENT SEMANTIC 5")
print(ls_5)

# RATING SEMANTICS
col_sums = factors[2].asnumpy().sum(axis=0)
factors[2] = factors[2].asnumpy()/col_sums[np.newaxis,:]

ls_1 = []
ls_2 = []
ls_3 = []
ls_4 = []
ls_5 = []
with open('tag_space_matrix/rating_dict.csv', mode='r') as infile:
    reader = csv.reader(infile)
    rating_dict = {rows[0]:rows[1] for rows in reader}
for i in range(len(factors[2])):
 row = factors[2][i]
 #print(row)
 num = np.ndarray.argmax(row)
 if num==0:
  ls_1.append(rating_dict[str(i)])
 if num==1:
  ls_2.append(rating_dict[str(i)])
 if num==2:
  ls_3.append(rating_dict[str(i)])
 if num==3:
  ls_4.append(rating_dict[str(i)])
 if num==4:
  ls_5.append(rating_dict[str(i)])

print("LATENT SEMANTIC 1")
print(ls_1)

print("LATENT SEMANTIC 2")
print(ls_2)

print("LATENT SEMANTIC 3")
print(ls_3)

print("LATENT SEMANTIC 4")
print(ls_4)

print("LATENT SEMANTIC 5")
print(ls_5)

