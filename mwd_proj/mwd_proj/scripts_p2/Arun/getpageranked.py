import csv
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import ppr

setActors = set([74])
#setActors = set([0,4,8,12,16])
results=[]
with open("tag_space_matrix/coactor_matrix.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
	temp=[]
	for element in row:
	  temp.append(float(element))
	results.append(temp)
#print(results)
#dist_out = 1-pairwise_distances(results, metric="cosine")

nodes,s=ppr.powerIteration(setActors,results)
#print(s)
print(s)
result = list(reversed(sorted(range(len(s)), key=lambda k: s[k])))
print(result[:10])


