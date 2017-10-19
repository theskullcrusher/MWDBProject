import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
from core import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA

mydb = mysql.connector.connect(host='localhost',
    user='root',
    passwd='padfoot',
    db='mwd')
cursor = mydb.cursor(buffered=True)

cursor.execute("select distinct(genre) from master");
ret = cursor.fetchall()
genres = [i[0] for i in ret]

cursor.execute("select distinct(tagid) from master")
ret = cursor.fetchall()
tags = [x[0] for x in ret]

'''Get actorid-actor names dict'''
actor_names = {}
cursor.execute('select *from imdb_actor_info')
ret = cursor.fetchall()
for x in ret:
    actor_names[x[0]] = x[1]


'''Here the data is (genre X actors) with each cell having Tf-IDF values for that genre and actor'''
def compute_Semantics_1b(method, genre):
    cursor.execute("select distinct(actorid) from master");
    ret = cursor.fetchall()
    actors = [i[0] for i in ret]

    if (genre not in genres):
        print "Genre doesn't exist in the dataset!"
        return

    '''Matrix Dataset'''
    V = sp.lil_matrix((len(genres), len(actors)))
    decomposed = []
    '''get tf-idfs vectors for each genre w.r.t actors'''
    count = 0
    for i in range(len(genres)):
        g = genres[i]
        # tf_idf = compute_tf_idf_movie(cur_movie,"TF-IDF")
        tf_idf = compute_tf_idf_actor_genre(g)
        '''
            cell = [0]
            if actors[j] in tf_idf.keys():
                cell = [a[1] for a in tf_idf if a[0] == actors[j]]
                # print "found tag",tags[i],": ",cell'''
        for j in range(len(actors)):
            V[i, j] = tf_idf[actors[j]]

    if (method == 'SVD'):
        '''  SVD  Calculation '''
        U, sigma, Vt = svds(V, k=4)
        sigma = np.diag(sigma)
        # print "\n\nSigma = \t",sigma
        print "\n\nU:", len(U), len(U[0]), "Sigma: ", sigma.shape, " V: ", Vt.shape, "\n\n"
        #print U
        #print "For genre Latent semantics are:", U[genres.index(genre)]
        decomposed = U
    if (method == 'PCA'):
        # standardizing data
        V = sp.csr_matrix(V).todense()
        V_std = StandardScaler().fit_transform(V)
        print "Stdandardized size: ", V_std.shape

        '''PCA::   Using Inbuilt library function'''
        sklearn_pca = PCA(n_components=4)
        pca = sklearn_pca.fit(V_std)
        Vt = pca.components_
        # print Vt
        decomposed = pca.transform(V_std)

    '''SVD,PCA :: IN order to give Latenet Semantics some names: Normalize each column in feature factor matrix
                      and then pick top 5 actors somewhat describing that Latent Semantic '''
    #normed_Vt = normalize(Vt, axis=0, norm='max')
    normed_Vt = Vt / Vt.sum(axis=0)
    # print "\n\nHo ho!!\n", normed_Vt
    for i in range(4):
        idx = np.argpartition(-normed_Vt[i], 5)[:5]
        # print "What is this?", -np.partition(-normed_Vt[0], 5)[:5]
        # print idx
        print "Latent Semantic: ", i + 1, " = "
        li = []
        for j in idx:
            li.append(actor_names[actors[j]])
        print '\t', li, "\n"

    return decomposed
def compute_Semantics_1a(method, genre,k_topics):
    if(genre not in genres):
        print "Genre doesn't exist in the dataset!"
        return
    '''Matrix Dataset'''
    V = sp.lil_matrix((len(genres), len(tags)))
    decomposed = []
    '''get tf-idfs vectors for genre-tag pairs and fill the matrix
        0 if genre-tag doesn't exist'''

    for i in range(len(genres)):
        g = genres[i]
        # tf_idf = compute_tf_idf_movie(cur_movie,"TF-IDF")
        tf_idf = compute_TASK2(g, "TF-IDF")
        for j in range(len(tags)):
            V[i, j] = tf_idf[tags[j]]

    if(method == 'SVD'):
        '''  SVD  Calculation '''
        U, sigma, Vt = svds(V, k=k_topics)
        sigma = np.diag(sigma)
        # print "\n\nSigma = \t",sigma
        print "\n\nU:", len(U), len(U[0]), "Sigma: ", sigma.shape, " V: ", Vt.shape, "\n\n"
        #print U
        decomposed = U
        print "For genre Latent semantics are:", U[genres.index(genre)]


    if(method == 'PCA'):
        # standardizing data
        V = sp.csr_matrix(V).todense()
        V_std = StandardScaler().fit_transform(V)
        #print "Stdandardized size: ", V_std.shape

        '''PCA::   Using Inbuilt library function'''
        sklearn_pca = PCA(n_components=k_topics)
        pca = sklearn_pca.fit(V_std)
        Vt = pca.components_
        #print Vt
        decomposed = pca.transform(V_std)

    if (method == 'LDA'):
        '''TO:DO://  Create matrix with doc as rows and words as column s with each cell having freq count not tf-idf'''
        for i in range(len(genres)):
            g = genres[i]
            cursor.execute("select tagid,weight from task2 where genre like %s", ('%' + g + '%',))
            ret = cursor.fetchall()
            tags_in_genre = [x[0] for x in ret]

            for j in range(len(tags)):
                cell = tags_in_genre.count(tags[j])
                V[i, j] = cell

        lda = LDA(n_components=k_topics, max_iter=10000, learning_method="batch")
        lda.fit(V)
        Vt = lda.components_
        decomposed = lda.transform(V)
        lda = LDA(n_components=k_topics, max_iter=200, learning_method="batch")
        print "200 iterations: \n",lda.fit_transform(V)
        #print decomposed.shape, "\n", decomposed, "\n\n\n"
    '''IN order to give Latenet Semantics some names: Normalize each column in feature factor matrix
                      and then pick top 5 tags somewhat describing that Latent Semantic '''
    #normed_Vt = normalize(Vt, axis=0, norm='max')
    normed_Vt = Vt/Vt.sum(axis=0)
    #print "\n\nHo ho!!\n", normed_Vt
    for i in range(k_topics):
        idx = np.argpartition(-normed_Vt[i], 5)[:5]
        # print "What is this?", -np.partition(-normed_Vt[0], 5)[:5]
        print idx
        print "Latent Semantic: ", i + 1, " = "
        li = []
        for j in idx:
            li.append(tag_names[tags[j]])
        print '\t', li, "\n"

    return decomposed
#compute_Semantics_1a("SVD","Action")
#print "SVD: \n",compute_Semantics_1a("SVD","Action",4)
#print compute_Semantics_1b("PCA","Action")
#print compute_Semantics_1a("LDA","Action",2)
print "LDA: \n",compute_Semantics_1a("LDA","Action",4)
mydb.commit()
cursor.close()
# #print V_std
# '''co-variance matrix'''
# mean_vec = np.mean(V_std, axis=0)
# #print "\nMean : \n",mean_vec
# #print "Mean vector: ",len(mean_vec),len(mean_vec[0]),mean_vec.shape
#
# cov_mat = np.cov(V_std.T.astype(float))
# cov_mat = (V_std - mean_vec).T.dot((V_std - mean_vec)) / (V_std.shape[0]-1)
# print "Covariance: ", len(cov_mat),len(cov_mat[0])
#
# eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# #print "Normal way: Eig vec = ",eig_vecs
#
# '''(Alternatively) fro PCA computation we can USE SVD as well by not doing covariance,eigenvector stuff by ourselves'''
# U,sigma,Vt = svds(V_std.T)
# #print "how abouth this? with SVD: \n",U