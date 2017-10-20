import csv
import os
import actor_TFIDF
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import ppr
from joblib import Parallel, delayed
import multiprocessing

filename = open('/home/arun/Desktop/mwdb/demo/phase1_dataset/imdb_actor_info.csv')
count = 0
reader = csv.reader(filename)

test_list = []
for row in reader:
 if count == 0:
	count+=1
	continue
 test_list.append(row[0])

alldata = []
print("before init")
actor_TFIDF.init()
print("init done")
'''
for row in reader:
	print(len(alldata))
	if count == 0:
		count = 1 
		continue
	result = actor_TFIDF.main(row[0])
	alldata.append(result)
print(len(alldata))
print(len(alldata[0]))
'''
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(actor_TFIDF.main)(row) for row in test_list)
actor_TFIDF.cleanup()

dist_out=[]
#dist_out = 1-pairwise_distances(results[0][:], metric="cosine")


with open("tag_space_matrix/actor_matrix.csv", "wb") as f:
   writer = csv.writer(f)
   writer.writerows(results)

setActors = set([0,1,2,3,4,5,6,7,8,9,10])
ppr_output=ppr.powerIteration(setActors,dist_out)
print(ppr_output)
print(sorted(ppr_output))
#with open("actor9_ppr.csv", "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(ppr_output)
