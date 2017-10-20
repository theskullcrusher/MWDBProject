import csv
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import sys

for_who = sys.argv[1]

results = []
with open("tag_space_matrix/"+for_who+".csv", "r") as f:
    reader = csv.reader(f)
    results = list(list(rec) for rec in csv.reader(f, delimiter=','))

dist_out = 1-pairwise_distances(results, metric="cosine")
for i in range(len(dist_out)):
	dist_out[i][i] =0.0
with open("similarity_matrix/"+for_who+".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(dist_out)

