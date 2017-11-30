# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 02:16:17 2017

@author: Bharadwaj
"""

import numpy as np
import pandas as pd
from random import *
import csv
import time

start_time = time.time()

def get_random_int(a, b, c):
    i = c
    count = 1 
    while i == c and count < 1000:
        i = randint(a, b)    
        count += 1
    return i


def compute_L_and_H(C, alpha_prime_j, alpha_prime_i, y_j, y_i):
    if(y_i != y_j):
        return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
    else:
        return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

def predict(X, w, b):
    return np.sign(np.dot(w.T, X.T) + b).astype(int)


def compute_eta(x_k, y_k, w, b):
    return predict(x_k, w, b) - y_k


def svm_train(X, y, max_passes, C=1.0, epsilon=0.001):
    global w
    global b
    
    n = X.shape[0]
    alpha = np.zeros((n))
    pass_count = 0
    
    while True:
        pass_count += 1
        alpha_prev = np.copy(alpha)
        for j in range(0, n):
            i = get_random_int(0, n-1, j)
            x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
            k_ij = np.dot(x_i, x_i.T) + np.dot(x_j, x_j.T) - 2 * np.dot(x_i, x_j.T)
            if k_ij == 0:
                continue
            
            alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
            (L, H) = compute_L_and_H(C, alpha_prime_j, alpha_prime_i, y_j, y_i)
            
            w = np.dot(alpha * y, X)
            b_tmp = y - np.dot(w.T, X.T)
            b = np.mean(b_tmp)
            
            Eta_i = compute_eta(x_i, y_i, w, b)
            Eta_j = compute_eta(x_j, y_j, w, b)
            
            # Set new alpha values
            alpha[j] = alpha_prime_j + float(y_j * (Eta_i - Eta_j))/k_ij
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)

            alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])
            
        # Check convergence
        diff = np.linalg.norm(alpha - alpha_prev)
        if diff < epsilon:
            break
        
        if pass_count >= max_passes:
            print("Exceeded maximum passes")
            return w,b
      
    b_tmp = y - np.dot(w.T, X.T)
    b = np.mean(b_tmp)
    w = np.dot(alpha * y, X)
    return w, b


def readData(filename, header=True):
    data, header = [], None
    index = 0
    with open(filename, "rt") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if index == 0:
                index += 1
                continue
            else:
                data.append(row)
    return (np.array(data), np.array(header))


(data, header) = readData("Train_data.csv", header=False)

df = pd.read_csv('Train_data.csv')
labels = list(set(list(data[:,-1].astype(int))))

w_dict = {}
b_dict = {}
for label in labels:
    X = data[:,4:-2].astype(float)
    y = []
    for index, row in df.iterrows():
        if row['label'] == label:
            y.append(1)
        else:
            y.append(-1)
    
    t_w, t_b = svm_train(X, y, 1000, 1.0, 0.001)
    w_dict[label] = t_w
    b_dict[label] = t_b
    
df_test_data = pd.read_csv('Test_data.csv')
df_test_data['label'] = 0
print("movieid,moviename,label")
for index, row in df_test_data.iterrows():
    for key in w_dict.keys():
        prediction_for_cur_label = predict(np.array([row['rating'], row['genre_tfidf'], row['year_norm'],
                                                     row['actor_tfidf']]), w_dict[key], b_dict[key])
        if prediction_for_cur_label >= 0:
            df_test_data.set_value(index, 'label', key)
            print(str(row['movieid']) +","+ str(row['moviename']) +","+ str(key))
            break

df_test_data.to_csv('Output_Part3.csv',index=False)

print("Done...")
print("--- %s seconds ---" % (time.time() - start_time))