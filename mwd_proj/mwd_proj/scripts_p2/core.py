import csv
import mysql.connector
from mysql.connector import errorcode
import datetime


import math

mydb = mysql.connector.connect(host='localhost',
    user='root',
    passwd='padfoot',
    db='MWD')

cursor = mydb.cursor(buffered=True)

tag_names = {}
cursor.execute('select *from genome_tags')
ret = cursor.fetchall()
for x in ret:
    tag_names[x[0]] = x[1]

#fetch tagnames

'''will be stored locally to reduce sql queries
    size will be 1 - max(rank)'''
rank_weights = []


'''Helper function for computing temporal weights of timestamps and adding it into database
    should be called before performing any tasks'''
def compute_tag_weights():

    query = "SELECT * FROM information_schema.COLUMNS  WHERE  TABLE_SCHEMA = 'mwd'  AND TABLE_NAME = 'mltags'  AND COLUMN_NAME = 'weight' "
    cursor.execute(query)
    ret = cursor.fetchall()
    if ret:
        print "Weights have already been computed"
        return

    query = "alter table mltags add weight float after timestamp"
    cursor.execute(query)
    #cursor.fetchall()

    cursor.execute("select min(timestamp) from mltags")
    min_t = cursor.fetchall()[0][0]
    cursor.execute("select max(timestamp) from mltags")
    max_t = cursor.fetchall()[0][0]

    cursor.execute("select timestamp from mltags")
    ret = cursor.fetchall()
    times = [x[0] for x in ret]

    divider = (max_t - min_t).total_seconds()*1000  #milisec

    for t in times:
        '''
            weight = (cur - min) / (max-min)
            This values will be in [0,1] we will scale it to [0.5,1] using ax+b
             a=0.5, b=0.5
         '''
        delta = (t - min_t).total_seconds()*1000

        weight = 0.5*(delta/divider)+0.5
        cursor.execute("update mltags set weight=%s where timestamp=%s",(weight,t))

def compute_rank_weights():
    cursor.execute("select min(actor_movie_rank) from movie_actor")
    min_rank = cursor.fetchall()[0][0]
    cursor.execute("select max(actor_movie_rank) from movie_actor")
    max_rank = cursor.fetchall()[0][0]

    divider = (max_rank - min_rank)
    rank_weights = range(1,max_rank+2)
    r = 1
    for r in range(1,max_rank+1):
        '''
            weight = (cur - min) / (max-min)
            This values will be in [0,1] we will scale it to [0.5,1] using ax+b
             But here rank 1 gets 0 and highest one gets 1 ...so we will reverse it by assigning
                weight 1 to rank 1 (more imp)  - weight 0.5 to highest rank
             a= -0.5, b=1
         '''
        delta = r - min_rank
        weight = -0.5*(delta/float(divider))+1
        #print("Rank: ",r," weight",weight)
        rank_weights[r] = weight
    return rank_weights

'''Helper function that returns total weight: based on tag and actor weight'''
def get_weight(tag_w,rank_w):
    '''tag and rank weights are in range [0.5 - 1]
        we will add them and convert this combined [1 - 2] range back into [0.5,1]
        a=0.5 b=0'''
    return (tag_w+rank_w)*0.5+0
''' Performs logic of task 1  '''

def init_master_table():
    query='create Table IF NOT EXISTS master select b.movieid,b.moviename,b.genre,b.tagid,b.timestamp,b.weight,a.actorid,a.actor_movie_rank FROM (select * from movie_actor ) as a JOIN  (select m.movieid,m.movieName,m.genre,t.tagid,t.timestamp,t.weight FROM mlmovies m JOIN mltags t ON m.movieid = t.movieid ) as b ON a.movieid = b.movieid'
    cursor.execute(query)
    #print 'Master table ready'

def compute_TASK1(actorid,model):
    init_master_table()
    cursor.execute("select tagid,weight,actor_movie_rank from master where actorid=%s", (actorid,))
    ret = cursor.fetchall()

    tags = [(x[0]) for x in ret]
    weights = [x[1] for x in ret]
    ranks = [x[2] for x in ret]
    # print 'tags for actor: ',tags
    set_tags = set(tags)
    no_tags = len(tags)
    sum_tags = sum(weights) #instead of total count , sum of all tag weights
    tfidf = []
    tf= []

    cursor.execute("select count(actorid) from imdb_actor_info")
    no_actors = cursor.fetchall()[0][0]
    #compute tf-idf for each tag found for this actor
    for t in set_tags:
        # print 'For tag :',t,"occurance count: ",tag_count[0][0]
        '''Instead of normal tf counting where no_of_t will be used
            we will use sum_weights instead to give more emphasis to newer tags'''
        no_of_t = tags.count(t)
        '''find tags and corresponding weights'''
        start = -1
        sum_weights = 0
        while True:
            try:
                i = tags.index(t, start + 1)
            except ValueError:
                break
            else:
                sum_weights+= get_weight(weights[i],rank_weights[ranks[i]])
                start = i
        tf_ = sum_weights / sum_tags
        tf.append((t,tag_names[t],tf_))
        if model == 'TF-IDF':
            cursor.execute("select count(DISTINCT actorid) from master where tagid=%s", (t,))
            actors_with_tag_t = cursor.fetchall()[0][0]
            idf = math.log(no_actors / actors_with_tag_t, 10)
            tf_idf = tf_ * idf;
            #print 'tag: ','tf:', tf_,' idf:',idf
            tfidf.append((t,tag_names[t],tf_idf))
    if model == 'TF':
        return tf
    else:
        return tfidf

def init_task2_table():
    query = "create Table IF NOT EXISTS task2 select m.movieid,m.tagid,m.userid,m.timestamp,m.weight,g.genre from mltags m JOIN mlmovies g ON m.movieid=g.movieid"
    cursor.execute(query)
    #print 'task2 is ready'

''' Performs logic of task 2  '''
def compute_TASK2(genre,model):
    init_task2_table()
    cursor.execute("select tagid,weight from task2 where genre like %s", ('%'+genre+'%',))
    ret = cursor.fetchall()

    tags = [x[0] for x in ret]
    set_tags = set(tags)
    no_tags = len(tags)

    weights = [x[1] for x in ret]
    sum_tags = sum(weights)  # instead of total count , sum of all tag weights

    #total no of different genres
    cursor.execute("select count(DISTINCT genre) from mlmovies")
    no_genres = cursor.fetchall()[0][0]
    tf = []
    tfidf = []

    for t in set_tags:
        ''' Distinct because same movie,genre has different tags we need to avoid duplicate enris in this context
            We need only no of genres which hav been assigned this particular tag
            run this query without distinct and you will get to know'''
        #print 'For tag :',t,"occurance count: ",genre_with_tag_t
        '''Instead of normal tf counting where no_of_t will be used
            we will use sum_weights instead to give more emphasis to newer tags'''

        '''find tags and corresponding weights'''
        start = -1
        sum_weights = 0
        while True:
            try:
                i = tags.index(t, start + 1)
            except ValueError:
                break
            else:
                sum_weights += weights[i]
                start = i

        #no_of_t = tags.count(t)
        #tf_n = no_of_t / float(no_tags)
        #print 'tag:', t, ' Normal sum: ', no_of_t, ' weighted: ', sum_weights
        tf_ = sum_weights / sum_tags
        tf.append((t,tag_names[t],tf_))
        if model == "TF-IDF":
            cursor.execute("select count(distinct genre) from task2 where tagid=%s", (t,))
            genre_with_tag_t = cursor.fetchall()[0][0]
            idf = math.log(no_genres / genre_with_tag_t, 10)
            tf_idf = tf_ * idf;
            #print '\t weighted tf:', tf,' idf:',idf,' tf*idf = ',tf_idf
            #print "tag: ",t,"  : ",tf_idf
            tfidf.append((t,tag_names[t], tf_idf))
    if model == 'TF':
        return tf
    else:
        return tfidf

def compute_tf_idf_actor_genre(genre):
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



def compute_tf_idf_movie(movieid,model):
    cursor.execute("select tagid,weight from master where movieid=%s", (movieid,))
    ret = cursor.fetchall()

    tags = [x[0] for x in ret]
    set_tags = set(tags)
    no_tags = len(tags)

    weights = [x[1] for x in ret]
    sum_tags = sum(weights)  # instead of total count , sum of all tag weights

    # total no of different genres
    cursor.execute("select count(DISTINCT movieid) from mlmovies")
    no_movies = cursor.fetchall()[0][0]
    tf = []
    tfidf = []
    for t in set_tags:
        '''find tags and corresponding weights'''
        start = -1
        sum_weights = 0
        while True:
            try:
                i = tags.index(t, start + 1)
            except ValueError:
                break
            else:
                sum_weights += weights[i]
                start = i

        tf_ = sum_weights / sum_tags
        tf.append((t,tag_names[t],tf_))
        if model == "TF-IDF":
            cursor.execute("select count(distinct movieid) from master where tagid=%s", (t,))
            movies_with_tag_t = cursor.fetchall()[0][0]
            idf = math.log(no_movies / movies_with_tag_t, 10)
            tf_idf = tf_ * idf;
            tfidf.append((t,tag_names[t], tf_idf))
    if model == 'TF':
        return tf
    else:
        return tfidf

''' Performs logic of task 1  '''
def compute_TASK3(userid,model):
    #cursor.execute("select tagid,weight from mltags where userid=%s", (userid,))
    #ret = cursor.fetchall()
    #tags = [x[0] for x in ret]
    #weights = [x[1] for x in ret]

    cursor.execute('select a.tagid, a.weight from  mltags a, mlratings b where b.userid =%s and a.movieid = b.movieid',(userid,))
    ret = cursor.fetchall()
    more_tags = [x[0] for x in ret]
    more_weights = [x[1] for x in ret]

    tags = more_tags
    weights = more_weights
    set_tags = set(tags)
    no_tags = len(tags)


    sum_tags = sum(weights)  # instead of total count , sum of all tag weights
    tag_weights = []

    #total no of different genres
    cursor.execute("select count(DISTINCT userid) from mlusers")
    no_users = cursor.fetchall()[0][0]
    tf = []
    tfidf = []

    for t in set_tags:
        #print 'For tag :',t,"users count: ",users_with_tag_t
        '''Instead of normal tf counting where no_of_t will be used
            we will use sum_weights instead to give more emphasis to newer tags'''

        '''find tags and corresponding weights'''
        start = -1
        #instead of count(no_of_t) Sum of temporal weights for tag t for this user
        sum_weights = 0
        while True:
            try:
                i = tags.index(t, start + 1)
            except ValueError:
                break
            else:
                sum_weights += weights[i]
                start = i

        no_of_t = tags.count(t)
        tf_n = no_of_t / float(no_tags)
        #print 'tag:', t, ' Normal sum: ', no_of_t, ' weighted: ', sum_weights
        tf_ = sum_weights / sum_tags
        tf.append((t, tag_names[t], tf_))
        if model == 'TF-IDF':
            #cursor.execute("select count(distinct userid) from mltags where tagid=%s", (t,))
            #users_with_tag_t = cursor.fetchall()[0][0]

            #select count(distinct b.userid) from mlratings b , mltags a  where tagid=%s and a.movieid=b.movieid
            cursor.execute("select count(distinct m.u) from  (select userid as u, movieid from mlratings where  movieid IN(select movieid from mltags where tagid = %s)) m", (t,))
            users_with_tag_t = cursor.fetchall()[0][0]
            idf = math.log(no_users / users_with_tag_t, 10)

            tf_idf = tf_ * idf;
            #print '\tnormal tf:', (tf_n), ' weighted tf:', tf
            tfidf.append((t,tag_names[t], tf_idf))
    if model == 'TF':
        return tf
    else:
        return tfidf
def compute_TASK4(genre1,genre2,model):
    init_task2_table()
    cursor.execute("select tagid from task2 where genre like %s", ('%' + genre1 + '%',))
    ret = cursor.fetchall()
    g1_tags = [x[0] for x in ret]
    cursor.execute("select tagid from task2 where genre like %s", ('%' + genre2 + '%',))
    ret = cursor.fetchall()
    g2_tags = [x[0] for x in ret]

    if model == 'TF-IDF-DIFF':
        compute_TF_IDF_DIFF(genre1,genre2,g1_tags,g2_tags)
    elif model == 'P-DIFF1':
        compute_P_DIFF1(genre1,genre2,g1_tags,g2_tags)
    elif model == 'P-DIFF2':
        compute_P_DIFF2(genre1, genre2, g1_tags, g2_tags)

def compute_P_DIFF1(genre1,genre2,g1_tags,g2_tags):
    cursor.execute("select distinct movieid from mlmovies where genre like %s", ('%' + genre1 + '%',))
    ret = cursor.fetchall()
    movies_g1= [x[0] for x  in ret]

    cursor.execute("select distinct movieid from mlmovies where genre like %s",('%' + genre2 + '%',))
    ret = cursor.fetchall()
    movies_g2 = [x[0] for x in ret]

    #compue weights for genre 1
    movies = set.union(set(movies_g1),set(movies_g2))
    M = len(movies)
    g1_weights = []
    g2_weights = []
    R_1 = len(movies_g1)
    R_2 = len(movies_g2)
    all_tags = set.union(set(g1_tags),set(g2_tags))
    for t in all_tags:
        cursor.execute("select count(distinct movieid) from task2 where genre like %s and tagid=%s", ('%' + genre1 + '%',t))
        r = cursor.fetchall()[0][0]
        cursor.execute("select count(distinct movieid) from task2 where (genre like %s or genre like %s) and (tagid=%s)", ('%' + genre1 + '%','%' + genre2 + '%', t))
        m = cursor.fetchall()[0][0]
        y=0
        x=0
        print  'TAG:',t,'r: ',r,' m:',m,' M:',M,' R_1:',R_1
        y = abs((float(r) / R_1) - (float(m - r) / (M - R_1)))
        if r==0 or r==m:
            #do smoothing
            prob_g1 = (float(r) + 0.5) / (R_1 + 1)
            prob_g2 = (float(m - r) + 0.5) / ((M - R_1) +1)
            x = ((prob_g1 * (1 - prob_g2)) / ((1 - prob_g1) * prob_g2))
            y = abs(prob_g1 - prob_g2)
        else:
            x = (float(r)/(R_1 - r))/( float(m-r)/ (M - m - R_1 + r))
        print x,' ',y
        weight = math.log(x,10)*y
        g1_weights.append((tag_names[t],weight))

        # compue weights for genre 2
        r = 0
        cursor.execute("select count(distinct movieid) from task2 where genre like %s and tagid=%s",
                       ('%' + genre2 + '%', t))
        r = cursor.fetchall()[0][0]
        y = abs((float(r) / R_2) - (float(m - r) / (M - R_2)))

        if r==0 or r==m:
            #do smoothing
            prob_g1 = (float(r) +0.5) / (R_2+1)
            prob_g2 = (float(m - r)+0.5) / ((M - R_2)+1)
            x = ((prob_g1 * (1 - prob_g2)) / ((1 - prob_g1) * prob_g2))
            y = abs(prob_g1 - prob_g2)
        else:
            x = (float(r)/(R_2 - r))/( float(m-r)/ (M - m - R_2 + r))
        weight = math.log(x,10)*y
        g2_weights.append((t,weight))
    print_diff(g1_weights,g2_weights)

def compute_P_DIFF2(genre1,genre2,g1_tags,g2_tags):
    cursor.execute("select distinct movieid from mlmovies where genre like %s", ('%' + genre1 + '%',))
    ret = cursor.fetchall()
    movies_g1= [x[0] for x  in ret]

    cursor.execute("select distinct movieid from mlmovies where genre like %s",('%' + genre2 + '%',))
    ret = cursor.fetchall()
    movies_g2 = [x[0] for x in ret]

    #compue weights for genre 1
    movies = set.union(set(movies_g1),set(movies_g2))
    M = len(movies)
    g1_weights = []
    g2_weights = []
    R_1 = len(movies_g1)
    R_2 = len(movies_g2)
    all_tags = set.union(set(g1_tags),set(g2_tags))
    for t in all_tags:
        cursor.execute("select count(distinct movieid) from task2 where genre like %s and tagid <> %s", ('%' + genre2 + '%',t))
        r = cursor.fetchall()[0][0]
        cursor.execute("select count(distinct movieid) from task2 where (genre like %s or genre like %s) and (tagid <> %s)", ('%' + genre1 + '%','%' + genre2 + '%', t))
        m = cursor.fetchall()[0][0]
        y=0
        x=0
        #print  'TAG:',t,'r: ',r,' m:',m,' M:',M,' R:',R_2
        y = abs((float(r) / R_2) - (float(m - r) / (M - R_2)))
        if r==0 or r==m:
            #do smoothing
            prob_g1 = (float(r) + 0.5) / (R_2 + 1)
            prob_g2 = (float(m - r) + 0.5) / ((M - R_2) +1)
            x = ((prob_g1 * (1 - prob_g2)) / ((1 - prob_g1) * prob_g2))
            y = abs(prob_g1 - prob_g2)
        else:
            x = (float(r) / (R_2 - r)) / (float(m - r) / (M - m - R_2 + r))

        weight = math.log(x,10)*y
        g1_weights.append((tag_names[t],weight))


        # compue weights for genre 2
        r = 0
        cursor.execute("select count(distinct movieid) from task2 where genre like %s and tagid <> %s",('%' + genre1 + '%', t))
        r = cursor.fetchall()[0][0]

        y = abs((float(r) / R_1) - (float(m - r) / (M - R_1)))
        if r == 0 or r==m:
            # do smoothing
            prob_g1 = (float(r) + 0.5) / (R_1 + 1)
            prob_g2 = (float(m - r) + 0.5) / ((M - R_1) + 1)
            x = ((prob_g1 * (1 - prob_g2)) / ((1 - prob_g1) * prob_g2))
            y = abs(prob_g1 - prob_g2)
        else:
            x = (float(r) / (R_1 - r)) / (float(m - r) / (M - m - R_1 + r))


        weight = math.log(x,10)*y
        g2_weights.append((tag_names[t],weight))

    print_diff(g1_weights,g2_weights)

def compute_TF_IDF_DIFF(genre1,genre2,g1_tags,g2_tags):
    g1_tf = {a:c for a,b,c in compute_TASK2(genre1,'TF')}
    g1_idf = []
    g2_tf = {a:c for a,b,c in compute_TASK2(genre2,'TF')}
    g2_idf = []

    set_g1 = set(g1_tags)
    set_g2 = set(g2_tags)

    #instead of considering all movies we are looking at just movies with g1,g2

    cursor.execute('select count(distinct genre,movieid) from task2 where (genre like %s or genre like %s)',('%'+genre1+'%','%'+genre2+'%',))
    no_of_genres = cursor.fetchall()[0][0]
    for t in set_g1:
        '''genre_with_tag_t = 1
        if t in set_g2:
            #means tag present in  both genre -> not unique to this genre IDF will ignore this
            genre_with_tag_t = 2
        '''
        #change it it to sum of all tag weights with g1,g2 / sum of tag weights with g1 with t
        cursor.execute('select count(distinct genre,movieid) from task2 where (tagid=%s and genre like %s)',(t,'%'+genre1+'%'))
        genre_with_tag_t = cursor.fetchall()[0][0]
        idf = math.log( no_of_genres / genre_with_tag_t, 10)
        tf_idf = g1_tf[t] * idf
        g1_idf.append((tag_names[t], tf_idf))

    #for such tags tf = 0 so tf*idf = 0
    for t in set_g2-set_g1:
        g1_idf.append((tag_names[t], 0))

    for t in set_g1-set_g2:
        g2_idf.append((tag_names[t], 0))

    for t in set_g2:

        cursor.execute('select count(distinct genre,movieid) from task2 where (tagid=%s and genre like %s)',(t,'%'+genre2+'%'))
        genre_with_tag_t = cursor.fetchall()[0][0]

        idf = math.log( no_of_genres / genre_with_tag_t, 10)
        tf_idf = g2_tf[t] * idf
        g2_idf.append((tag_names[t], tf_idf))
    print print_diff(g1_idf,g2_idf)


def print_diff(g1,g2):
    dif = []
    for x,y in zip(g1,g2):
        diff = x[1] - y[1]
        dif.append((x[0],diff))

    dif.sort(key=lambda x: x[1], reverse=True)

    for d in dif:
        print d

compute_tag_weights()
rank_weights = compute_rank_weights()


