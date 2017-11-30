import pandas as pd
import math
import operator

average_diff = 0.0

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(int(length)):
        if x!=1:
            distance += pow((instance1[x] - instance2[x]), 2)
        else:
            #print instance1[x], instance2[x]
            split_genre_1 = list(instance1[x].split("|"))
            split_genre_2 = list(instance2[x].split("|"))
            set_1 = set(split_genre_1 + split_genre_2)
            term = float(len(set_1)/len(split_genre_1+split_genre_2))
            value = float(term)*float(average_diff)
            #print len(set_1), len(split_genre_1+split_genre_2), term, value, average_diff
            distance += (value*value)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	features = len(testInstance)
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], features)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

old_df = pd.read_csv('Train_data.csv')
old_df_Test = pd.read_csv('Test_data.csv')

k = int(input())
df = old_df[['movieid','year','genres','rating','timestamp','label']]
num_label_samples = len(df)
num_features = len(df.columns)-2

df_Test = old_df_Test[['movieid','year','genres','rating','timestamp']]
num_unlabel_samples = len(df_Test)

label_data = [[0 for i in range(num_features+1)] for j in range(num_label_samples)]
unlabel_data = [[0 for i in range(num_features+1)] for j in range(num_unlabel_samples)]
output_data = [[0 for i in range(3)] for j in range(num_unlabel_samples)]

i = 0
for index,row in df.iterrows():
    temp_list = []
    for j in range(1,len(df.columns)):
        temp_list.append(row[j])
    label_data[i] = temp_list
    i+=1

i = 0
for index,row in old_df_Test.iterrows():
    temp_list = []
    output_data[i][0] = int(row[0]) 
    output_data[i][1] = row[8] 
    for j in range(1,len(df_Test.columns)):
        temp_list.append(row[j])
    unlabel_data[i] = temp_list
    i+=1

for i in range(1,len(df.columns)-1):
    if i!=2:
        particular_group = list(df.iloc[:,i])
        average_diff += max(particular_group) - min(particular_group)
average_diff = float(average_diff)/float((num_features-1))
if average_diff == 0:
    average_diff = 1

predictions = []

for i in range(len(unlabel_data)):
    neighbors = getNeighbors(label_data, unlabel_data[i], k)
    result = getResponse(neighbors)
    predictions.append(result)
    
for i in range(num_unlabel_samples):
    output_data[i][2] = predictions[i]
    
df_out = pd.DataFrame(data=output_data,index=None, columns=None)
df_out.to_csv('Output_Part1.csv',header=['MovieId','MovieName','Label'],index=False)