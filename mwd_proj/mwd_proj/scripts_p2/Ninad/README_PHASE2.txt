READ-ME

final demo will use these Py files:

PRE - COMPUTATION
matrix_factorization.py (for generating the user_movie_rating dataframe)
matrix_factorization_comp.py (for generating the reconstructed user-movie_rating )

To Get recommendations.
matrix_factorization_recommender.py (for giving the recommendations)


Usage:
1.MODEL 1 (Using content based recommendation i.e recommending movies based on most watched genre):
=====================================================================================================
a.Create a database to use and update it in mysqlConn.py

b.Import the csv datafiles to the created database as follows : 

ninad@ninad-lpc:~$ python queryDB.py "/home/ninad/ASU_DATA/Courses/MWD/Project/Phase 2/Phase2_data"

Importing data from path :  /home/ninad/ASU_DATA/Courses/MWD/Project/Phase 2/Phase2_data
imdb-actor-info - done
genome-tags - done
mlmovies - done
mlratings done
mltags done
mlusers done
movie-actor done
All data has been Imported

c.Pass user for which recommendation is required.
ninad@ninad-lpc:~$ python user_movie_matrix.py 1027

-----Watched movies-------
('Pitch Black', 'Thriller')
('Romeo Must Die', 'Crime|Action|Thriller')
('U-571', 'Action|Thriller')
('Dawn of the Dead', 'Action|Thriller|Drama')
('Constantine', 'Action|Thriller')
('Cloverfield', 'Action|Thriller')
----------------------------------------------------------------------
-------Genre weights based in user rating of movie and movie year-----
[('Thriller', 9.435909),
 ('Action', 9.329804),
 ('Crime', 1.3479869999999998),
 ('Drama', 0.898658),
 ('Adventure', 1e-06)]
-------------------------------
-------Recommended movies------
('Scream 3', 'Thriller')
('Trois', 'Thriller')
('Antitrust', 'Thriller')
('Hannibal', 'Thriller')
('3000 Miles to Graceland', 'Action|Thriller')


2.MODEL-2 (Some imporvements to the previous model)
========================================================================
a.Create a database to use and update it in mysqlConn.py

b.Import the csv datafiles to the created database as follows : 

ninad@ninad-lpc:~$ python queryDB.py "/home/ninad/ASU_DATA/Courses/MWD/Project/Phase 2/Phase2_data"

Importing data from path :  /home/ninad/ASU_DATA/Courses/MWD/Project/Phase 2/Phase2_data
imdb-actor-info - done
genome-tags - done
mlmovies - done
mlratings done
mltags done
mlusers done
movie-actor done
All data has been Imported

c.ninad@ninad-lpc:~$ python user_movie_matrix2.py 1027
General preprocessing done.
Starting preprocessing for genre tag vectors...
done
-----Watched movies-------
('Pitch Black', 'Thriller')
('Romeo Must Die', 'Crime|Action|Thriller')
('U-571', 'Action|Thriller')
('Dawn of the Dead', 'Action|Thriller|Drama')
('Constantine', 'Action|Thriller')
('Cloverfield', 'Action|Thriller')
-------------------------------------
-------Top 5 Recommended movies------
('Dungeons & Dragons', 'Adventure')
('Bigger Than the Sky', 'Drama')
('Robots', 'Adventure')
('Because of Winn-Dixie', 'Drama')
('Final Destination 2', 'Thriller')


3.MODEL 3 (Using latent factors for movie recommendation)
============================================================================
a.USE SAME database as used in MODEL-2

b.Generate the user-movie_rating matrix
ninad@ninad-lpc:~$ python matrix_factorization.py 

c.Generate the Predicted user-movie_rating matrix - SVD and ALS
------------------
117.000162003
0th iteration is completed
79.97045607
1th iteration is completed
.
.
.
1.55111463462
48th iteration is completed
1.50961125538
49th iteration is completed

d.Get the recommendation. (Notet- Observed that this data varies with the size of the input)
ninad@ninad-lpc:~/ASU_DATA/Courses/MWD/Project/Phase 2/phase2Code$ python matrix_factorization_recommender.py 1027
-----Watched movies-------
('Pitch Black', 'Thriller')
('Romeo Must Die', 'Crime|Action|Thriller')
('U-571', 'Action|Thriller')
('Dawn of the Dead', 'Action|Thriller|Drama')
('Constantine', 'Action|Thriller')
('Cloverfield', 'Action|Thriller')
-------Top 5 Recommended movies------
('4354', 4.0)
('Final Fantasy: The Spirits Within', 'Adventure')
('6057', 4.0)
('Shanghai Knights', 'Action|Adventure')
('3323', 3.0)
('Erin Brockovich', 'Drama')
('3961', 3.0)
('Antitrust', 'Thriller')
('8939', 3.0)
('Robots', 'Adventure')

