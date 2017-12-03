#=======================================================
#Import packages
#=======================================================
import sys, os
from datetime import datetime
import django
import traceback
os.environ['DJANGO_SETTINGS_MODULE']="mwd_proj.settings"
django.setup()
from mwd_proj.utils.utils2 import *
import traceback
from django.db.models import Sum
import operator
import math
from django.db.models.functions import Lower
from mwd_proj.phase2.models import *
from django.db.models import Q
from mwd_proj.scripts_p2 import (print_genreactor_vector, print_genre_vector, print_user_vector, print_actor_vector, print_movie_vector)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
from math import log,exp
import pprint
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation as LDA
from numpy.linalg import *
from scipy.stats.stats import pearsonr
from numpy import linalg as la
from scipy.sparse.linalg import svds
matplotlib.style.use('ggplot')

#============================================================================
#Get input data matrix
#============================================================================
#usr_mvrating_matrix = get_user_mvrating_DF()
# usr_genre_matrix = usr_genre_matrix.T
# pprint.pprint(usr_genre_matrix)
#usr_mvrating_matrix.to_csv("factorization_1_user_mvrating.csv", sep='\t')

def preprocessor():
    # load data points
    with open("factorization_1_user_mvrating.csv") as f:
        ncols = len(f.readline().split('\t'))

    #R = pd.DataFrame(loadtxt('factorization_1_user_mvrating.csv',delimiter='\t', skiprows=1, usecols=range(1,ncols)))
    R = loadtxt('factorization_1_user_mvrating.csv',delimiter='\t', skiprows=1, usecols=range(1,ncols))


    #kvm known_value_matrix = for values in R[i,j] >0, kvm[i,j] = 1
    kvm = R>0.5
    kvm[kvm == True] = 1
    kvm[kvm == False] = 0
    # To be consistent with our R matrix
    kvm = kvm.astype(float64, copy=False)

    #print kvm
    #latent_sem = 86

    #=================================================================================
    #Calculate the SVD
    #The k value is 86 as selected by the packge by default,as number of movies is 86
    #=================================================================================

    #How can I select a reduced number of latent semantics here?
    k_topics = 50
    lda = LDA(n_components=k_topics, max_iter=10000, learning_method="batch",evaluate_every=10,perp_tol=1e-12)
    lda.fit(R)
    P = lda.components_
    Q = lda.transform(R)
    #s = linalg.svd(raw_data, full_matrices=False, compute_uv = False)

    #U_df = pd.DataFrame(U)
    #s_df = pd.DataFrame(s)
    #V_df = pd.DataFrame(V)


    #===============================================================================================
    #Read data from CSV gives error while reconstruction and convert to dataframe. So no need to compute the SVD for each program run.
    #This way we get the reconstruction matrix.
    #===============================================================================================

    #print U.shape, s.shape, V.shape

    #lb = regularization constant.
    ld=0.1
    #reg_value = ld*(sum(Q**2) + sum(P**2))

    weighted_errors = []
    n_iterations = 50 #After checking for 10,20,100 iterations, found that errors start to converge after 50 interations

    #print Q.shape
    #print P.shape
    #R_df = pd.DataFrame(R)

    #print "------------------"
    '''
    for ii in range(n_iterations):
        for u, Wu in enumerate(kvm):
            #print Wu
            #print diag(Wu)
            Q[u] = linalg.solve(dot(P, dot(diag(Wu), P.T)) + ld * eye(k_topics),dot(P, dot(diag(Wu), R[u].T))).T
            #Q[u] = linalg.solve(dot(P,P.T) + ld * eye(latent_sem), dot(P,R[u].T)).T

        for i, Wi in enumerate(kvm.T):
            P[:, i] = linalg.solve(dot(Q.T, dot(diag(Wi), Q)) + ld * eye(k_topics), dot(Q.T, dot(diag(Wi), R[:, i])))
            #P[:, i] = linalg.solve(dot(Q.T, Q) + ld * eye(latent_sem), dot(Q.T, R[:, i]))


        R_prime = dot(Q, P)

        MSE = sum((kvm * (R - R_prime)) ** 2)
        print "Error = ", MSE
        weighted_errors.append(MSE)
        #weighted_errors.append(sum((kvm * (R - R_prime))**2))
        #print weighted_errors
        print('{}th iteration is completed'.format(ii))
    '''
    weighted_R_hat = dot(Q, P)
    weighted_R_df = pd.DataFrame(weighted_R_hat);
    weighted_R_df.to_csv("R_final_lda.csv",sep='\t')


if __name__ == "__main__":
    preprocessor()
    print "Generates R_final ratings after applying SVD and ALS"
