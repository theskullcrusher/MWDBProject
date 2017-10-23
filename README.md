# MWDBProject

MWDB course Project Phase 2

0. Phase2 uses Django1.8 along with its ORM to manage the mysql database

1. Installing requirements(Only needed if you are setting up the repository):
 Run bash_setup_script.sh file in root directory of code using 
     >sh ./bash_setup_script.sh
 Run 
     > cd MWDBProject/mwd_proj
     > pip install -r requirements.txt
 Setup django project as python package:
     > cd MWDBProject/mwd_proj
     > export DJANGO_SETTINGS_MODULE=mwd_proj.settings
     > python setup.py develop
 Now you can you ipython to import the package and run the scripts too
     > ipython
        >> import django
        >> django.setup()
        >> from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector, print_movie_vector)
        >> from mwd_proj.scripts_p2 import (part1)
        >> from mwd_proj.scripts_p2.Arun import (part2, part3)
        >> from mwd_proj.scripts_p2.Ninad import (part4)
        >> from mwd_proj.phase2.models import *
        >> gt = GenomeTags.objects.get(tagid=13) #write django queries instead of sql


2-a. Setting up database:
 Create or change root user for MySQl with password: surajshah
 Access mysql from terminal as : mysql -u root -p surajshah
 Create a database called mwdb_suraj using the command:
 mysql> create database mwdb_suraj;
 mysql> use mwdb_suraj;
 Restore the mysql dump using the following command from root directory of code where the dump is stored:
 > cd MWDBProject
 > mysql -u root -p surajshah mwdb_suraj
 > source mwdb_suraj.sql


2-b. This is required to setup database from scratch on a different dataset and run all proprocessing scripts to populate
meta-data tables. This step is ORed with 2-a, based on the need.
> cd MWDBProject/mwd_proj/mwd_proj/scripts
> python populate_db.py
> cd MWDBProject/mwd_proj/mwd_proj/utils
> python utils.py
> cd MWDBProject/mwd_proj/mwd_proj/scripts_p2
Now here all tasks1 to task4 have a method called tf() which is commented in the __name__=="__main__" call because it needs to be executed only once to populate meta-data. Remove that commented call from all 4 task and just run the tf() methods once and comment the call to them again.


3. Instructions for testing all 4 tasks:
--> Open terminal export django settings and log into ipython
> export DJANGO_SETTINGS_MODULE=mwd_proj.settings
> ipython
        >> import django
        >> django.setup()
        >> from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector, print_movie_vector)
        >> from mwd_proj.scripts_p2 import (part1)
        >> from mwd_proj.scripts_p2.Arun import (part2, part3)
        >> from mwd_proj.scripts_p2.Ninad import (part4)
        >> from mwd_proj.phase2.models import *

        #Note - print the output variables to get decomposed matrices

        Task1:

        1a) a = part1.compute_Semantics_1a('<method>','<genre>',<k>)
            where: <method> = SVD, PCA or LDA
                   <genre> = Action, Horror, etc
                   <k> = No of latent_semantics to be considered
            eg: a=part1.compute_Semantics_1a('SVD','Action',4)

        1b) b = part1.compute_Semantics_1b('<method>','<genre>',<k>)
            where: <method> = SVD, PCA or LDA
                   <genre> = Action, Horror, etc
                   <k> = No of latent_semantics to be considered
            eg: b=part1.compute_Semantics_1b('SVD','Action',4)

        1c) c, z = part1.compute_Semantics_1c('<method>','<actor-name>','<similarity-measure>',<actors-count>,<k>,<flag>)
            where:  <method> = TF-IDF, SVD
                    <actor-name> = Lillard, Matthew; Affleck, Ben; etc
                    <similarity-measure> = cosine, euclidean
                    <actors-count> = No of similar actors to be returned
                    <k> = No of latent semantics to be considered for computations
                    <flag> = True, False - displays output on stdout if true
            eg: c, z = part1.compute_Semantics_1c('TF-IDF','Lillard, Matthew','cosine',10,5,True)
            eg: c, z = part1.compute_Semantics_1c('SVD','Lillard, Matthew','cosine',10,5,True)

       1d) d = part1.compute_Semantics_1d('<method>','<movie-name>',<actors-count>,<k>,<flag>)
            where:  <method> = TF-IDF, SVD
                    <movie-name> = Swordfish, Harry Potter and the Prisoner of Azkaban, Pitch Black, etc
                    <actors-count> = No of similar actors to be returned
                    <k> = No of latent semantics to be considered for computations
                    <flag> = True, False - displays output on stdout if true
            eg: d = part1.compute_Semantics_1d('SVD','Swordfish',10,5,True)

        Task2:

        2a) a = part2.compute_Semantics_2a(<k>,<max-actors>)
            where:  <k> = No of latent semantics
                    <max-actors> = maximum actors to output in each group
            eg: a = part2.compute_Semantics_2a(3,5)

        2b) b = part2.compute_Semantics_2b(<k>,<max-actors>)
            where:  <k> = No of latent semantics
                    <max-actors> = maximum actors to output in each group
            eg: b = part2.compute_Semantics_2b(3,5)

        2c) eg: c = part2.compute_Semantics_2c()

        2d) eg: d = part2.compute_Semantics_2d()

        Task3:

        3a) part3.compute_Semantics_3a(<actor-ids-list>)
            where:  <actor-ids-list> = set([1860883,486691,1335137,901175]), etc
            eg: part3.compute_Semantics_3a(set([1860883,486691,1335137,901175]))

        3b) part3.compute_Semantics_3b(<actor-ids-list>)
            where:  <actor-ids-list> = set([1860883,486691,1335137,901175]), etc
            eg: part3.compute_Semantics_3b(set([1860883,486691,1335137,901175]))


        Task4:

        #Please note this task requires following preprocessing to be run only once:
            > cd MWDBProject/mwd_proj/mwd_proj/scripts_p2/Ninad
            > python preprocessor_model3.py
            > python preprocessor_model3_comp.py

        4a) part4.compute_Semantics_4(<user-id>)
            where: <user-id> = 1027, etc
            eg: part4.compute_Semantics_4(1027)



MWDB course Project Phase 1

0. Phase1 uses Django1.8 along with its ORM to manage the mysql database

1. Installing requirements(Only needed if you are setting up the repository):
 Run install_packages.sh file in root directory of code using 
     >sh ./install_packages.sh
 Run 
     > cd MWDBProject/mwd_proj
     > pip install -r requirements.txt
 Setup django project as python package:
     > cd MWDBProject/mwd_proj
     > export DJANGO_SETTINGS_MODULE=mwd_proj.settings
     > python setup.py develop
 Now you can you ipython to import the package and run the scripts too
     > ipython
        >> import django
        >> django.setup()
        >> from mwd_proj.scripts.print_actor_vector import *
        >> from mwd_proj.phase1.models import *
        >> gt = GenomeTags.objects.get(tagid=13) #write django queries instead of sql

2-a. Setting up database:
 Create or change root user for MySQl with password: surajshah
 Access mysql from terminal as : mysql -u root -p surajshah
 Create a database called mwdb_suraj using the command:
 mysql> create database mwdb_suraj;
 mysql> use mwdb_suraj;
 Restore the mysql dump using the following command from root directory of code where the dump is stored:
 > cd MWDBProject
 > mysql -u root -p surajshah mwdb_suraj
 > source mwdb_suraj.sql


2-b. This is required to setup database from scratch on a different dataset and run all proprocessing scripts to populate
meta-data tables. This step is ORed with 2-a, based on the need.
> cd MWDBProject/mwd_proj/mwd_proj/scripts
> python populate_db.py
> cd MWDBProject/mwd_proj/mwd_proj/utils
> python utils.py
> cd MWDBProject/mwd_proj/mwd_proj/scripts
Now here all tasks1 to task4 have a method called tf() which is commented in the __name__=="__main__" call because it needs to be executed only once to populate meta-data. Remove that commented call from all 4 task and just run the tf() methods once and comment the call to them again.


3. Instructions for testing all 4 tasks:
Task1: To test task 1 run the command: (in the MWDBProject/mwd_proj/med_proj/scripts directory)
<actor-id> eg: 1575755, 506840, 
<model> eg: TF, TF-IDF

> python print_actor_vector.py <actor-id> <model>  #eg:  python print_actor_vector.py 506840 TF


Task2: To test task 2 run the command: (in the MWDBProject/mwd_proj/med_proj/scripts directory)
<genre> eg: Action, Documentary
<model> eg: TF, TF-IDF

> python print_genre_vector.py <genre> <model>  #eg:  python print_genre_vector.py Documentary TF-IDF


Task3: To test task 3 run the command: (in the MWDBProject/mwd_proj/med_proj/scripts directory)
<userid> eg: 146, 9316, 1988, 30167
<model> eg: TF, TF-IDF

> python print_user_vector.py <userid> <model>  #eg:  python print_user_vector.py 1988 TF


Task4: To test task 4 run the command: (in the MWDBProject/mwd_proj/med_proj/scripts directory)
<genre> eg: Action, Documentary, Horror
<model> eg: TF-IDF-DIFF, P-DIFF1, P-DIFF2

> python differentiate_genre.py <genre1> <genre2> <model>  #eg:  python differentiate_genre.py Action Drama P-DIFF2

#NOTE: Please note that if you run tf() method along with each task, it will take longer to execute. Please comment the call to this method if removed