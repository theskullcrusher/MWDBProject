import pandas as pd

def make_groups(index, value, dataset):
    left, right = list(), list()
    #print index
    if index==1:
        #this stands for genre
        for row in dataset:
            split_genre_1 = row[index].split("|")
            split_genre_2 = value.split("|")
            for key in split_genre_1:
                if key in split_genre_2:
                    left.append(row)
                else:
                    right.append(row)
        return left, right
    else:
        for row in dataset:
            #print row[index], index
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
 
def calc_gini_index(groups,class_labels):
    total_samples = float(sum([len(groups) for group in groups]))
    gini = 0.0
    for group in groups:
        size_group = float(len(group))
        if size_group == 0:
            continue
        score = 0.0
        for each_label in class_labels:
            probability_list = [row[-1] for row in group].count(each_label)/size_group
            score += probability_list*probability_list
        gini += (1.0 - score)*(size_group/total_samples)
    return gini

def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = make_groups(index, row[index], dataset)
			gini = calc_gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
def to_terminal(group):
	all_labels = [row[-1] for row in group]
	return max(set(all_labels), key=all_labels.count)
 
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

def predict(node, row):
    if node['index']==1:
        split_genre_1 = row[node['index']].split("|")
        split_genre_2 = node['value'].split("|")
        for key in split_genre_1:
            if key in split_genre_2:
                if isinstance(node['left'], dict):
                    return predict(node['left'], row)
                else:
                    return node['left']
            else:
                if isinstance(node['right'], dict):
                    return predict(node['right'], row)
                else:
                    return node['right']
    else:
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']

def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

arg_mini_size = 1
#int(input())

old_df = pd.read_csv('Train_data.csv')
old_df_Test = pd.read_csv('Test_data.csv')

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
#tree = build_tree(label_data,num_features-1,1)

answer_list = decision_tree(label_data,unlabel_data,num_features,arg_mini_size)

for i in range(num_unlabel_samples):
    output_data[i][2] = answer_list[i]

df_out = pd.DataFrame(data=output_data,index=None, columns=None)
df_out.to_csv('Output_Part2.csv',header=['MovieId','MovieName','Label'],index=False)
