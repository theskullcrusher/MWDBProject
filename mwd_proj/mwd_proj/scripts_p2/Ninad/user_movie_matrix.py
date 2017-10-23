'''
Generating a user-genre matrix.
Ranking the genres of a movie based on
    sum of movies rankings of a genre * year_wt of a movie (assumptiom : movies watched recently shows the latest choices of a user )
Based on the ranking of genres,
    recommend 5 unwatched movies total starting from the genres that are higher ranked.
'''


from mysqlConn import DbConnect
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
import operator
from math import log,exp
import pprint

#DB connector and curosor
db = DbConnect()
db_conn = db.get_connection()
cur2 = db_conn.cursor();

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("USER")
args = parser.parse_args()


#==========================================================
#TASK - 1 : PRE - PROCESSING
#==========================================================

#SUB -TASK 1 - Cleaning the mlmovies table. Getting single row for a single genre.

    #a. Create a new table mlmovies_clean that has a single entry for a single genre.
    #b. For each entry in the mlmovies create an entry in mlmovies_clean that has a unique genre entry.

cur2.execute("create table `mlmovies_clean`(movieid varchar(10) NOT NULL, moviename varchar(200) NOT NULL, year varchar(4) NOT NULL, genres varchar(200) NOT NULL)")

query1 = "SELECT * FROM `mlmovies`"
cur2.execute(query1)
result1 = cur2.fetchall()
row_count = 0

#For each tagID get the movie list
for entry in result1:
    mvid = entry[0]
    mvname = entry[1]
    year = entry[2]
    combo_genres = entry[3].split("|")

    #Add new row for each genre.
    for genre in combo_genres:
        cur2.execute('INSERT INTO `mlmovies_clean`(movieid, moviename, year, genres) VALUES(%s, %s, %s, %s)', (mvid,mvname,year,genre))

    if row_count >= 1000:
        db_conn.commit()
        row_count = 0

db_conn.commit()
#----------------------------------------------------------------------

#====================================================================
#TASK - 2 : Weights of movies based on year using exponential decay
#====================================================================

#Get the max year.
cur2.execute("SELECT max(year) FROM mlmovies")
max_year = int(cur2.fetchone()[0])

#add a column year_weight in the table mlmovies.
cur2.execute("Alter table `mlmovies` add column year_wt FLOAT(15) NOT NULL")

cur2.execute("SELECT year FROM `mlmovies`")
result = cur2.fetchall()

    # k = decay constant. Appropriate decay constant is used so that the exponential
    #  values stay within  limit (after trying with 1,0.1,0.01,0.001) and

k=0.1

for movie_year in result:
    current_year   = int(movie_year[0])
    diff = max_year - current_year
    movie_wt = float(exp(-k*diff))
    cur2.execute("UPDATE `mlmovies` set year_wt = %s where year = %s",(movie_wt,movie_year[0]))
    db_conn.commit()


#====================================================================
#Task-3 :Calculate the user_genre matrix
#====================================================================

dd_users_genre = {}


#Get all the users
#cur2.execute("SELECT userid FROM `mlusers` limit 500")

#Test all the 22000 users.
cur2.execute("SELECT userid FROM `mlusers`")
result0 = cur2.fetchall();
for usr in result0:
    #print usr[0]
    dd_users_genre[usr[0]] = {}

    #Get all movies watched(and hence rated) by each user.
    cur2.execute("SELECT movieid, rating FROM `mlratings` where userid = %s",usr)
    result1 = cur2.fetchall()
    for data1 in result1:
        #print data1
        user_movie_id = {data1[0],}
        user_movie_rating = data1[1]

        #for each movie add the genre weight.IN this way, genres will be ranked based on highest watched to
        #lowest watched for a user. Movies tagged or rated newly will have higher rank as year_wt is more.
        #But if movie rating is bad, then rank will drop as we multiply both year and rating

        cur2.execute("SELECT genres FROM `mlmovies_clean` where movieid = %s", user_movie_id)
        result2 = cur2.fetchall()
        for vals in result2:
            #print vals
            cur2.execute("SELECT year_wt FROM `mlmovies` where genres = %s", vals)
            mv_weight = cur2.fetchone()[0]

            if vals[0] in dd_users_genre[usr[0]]:
                dd_users_genre[usr[0]][vals[0]] += (mv_weight * int(user_movie_rating))
            else:
                dd_users_genre[usr[0]][vals[0]] = mv_weight * int(user_movie_rating)

    #WE need to do this again for mltags because it does not have a rating, so we give an avg below
    # rating of 2.

    # Get all movies tagged by each user. If movie is only tagged and not rated, then give rating of 2 (avg).
    cur2.execute("SELECT movieid FROM `mltags` where userid = %s", usr)
    result2 = cur2.fetchall()
    for data in result2:
        #print data1
        user_movie_id = {data[0],}

        cur2.execute("SELECT genres FROM `mlmovies_clean` where movieid = %s", user_movie_id)
        result2 = cur2.fetchall()
        for vals in result2:

            cur2.execute("SELECT year_wt FROM `mlmovies` where genres = %s", vals)
            mv_weight = cur2.fetchone()[0]

            if vals[0] in dd_users_genre[usr[0]]:
                dd_users_genre[usr[0]][vals[0]] += (mv_weight * 2)
            else:
                dd_users_genre[usr[0]][vals[0]] = mv_weight * 2


    #Make rating of other genres to verylow.
    cur2.execute("SELECT DISTINCT genres FROM `mlmovies_clean`")
    genreNames = cur2.fetchall()

    for keyval in genreNames:
        key = keyval[0]
        #print key
        if key in dd_users_genre[usr[0]]:
            continue
        else:
            dd_users_genre[usr[0]][key] = 0.000001



#pprint.pprint(dd_users_genre)
usr_genre_matrix = pd.DataFrame(dd_users_genre)
#usr_genre_matrix = usr_genre_matrix.T
#pprint.pprint(usr_genre_matrix)
usr_genre_matrix.to_csv("out.csv", sep='\t')

#========================================================================================
#Task:4 - Recommend top 5 unwatched movies starting from the best ranked genre for a user
#=========================================================================================


#This data can also be precomputed and stored
userWatchedMovies = []
cur2.execute("SELECT movieid FROM `mlratings` where userid = %s",[args.USER])
result0 = cur2.fetchall()
for data in result0:
    userWatchedMovies.append(data[0])

cur2.execute("SELECT movieid FROM `mltags` where userid = %s",[args.USER])
result0 = cur2.fetchall()
for data in result0:
    userWatchedMovies.append(data[0])

print "-----Watched movies-------"
for watched_ids in userWatchedMovies:
    cur2.execute("SELECT moviename,genres FROM `mlmovies` where movieid = %s", {watched_ids, })
    print cur2.fetchone()


user_genres_vals = sorted(list([usr_genre_matrix[args.USER]]),key=operator.itemgetter(1), reverse=True)
#print user_genres_vals

print "----------------------------------------------------------------------"
print "-------Genre weights based in user rating of movie and movie year-----"
sorted_x = sorted(dd_users_genre[args.USER].items(), key=operator.itemgetter(1), reverse=True)
pprint.pprint(sorted_x)

count=0
recommend=[]

print "-------------------------------"
print "-------Recommended movies------"

for keys in sorted_x:
    #print keys[0]
    cur2.execute("SELECT movieid FROM `mlmovies_clean` where genres = %s", {keys[0],})
    result0 = cur2.fetchall()
    for data in result0:
        if data[0] in userWatchedMovies:
            continue
        else:
            recommend.append(data[0])
            count+=1
            if count==5 : break
    if count == 5: break

for rec_ids in recommend:
    #print rec_ids
    cur2.execute("SELECT moviename,genres FROM `mlmovies` where movieid = %s", {rec_ids, })
    print cur2.fetchone()


