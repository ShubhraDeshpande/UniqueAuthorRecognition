from numpy import genfromtxt
import csv
import numpy as np 
import pickle
import random
import pickle


random.seed(0) #same shuffling everytime
num_features = 9

full_data_dict = {}
full_data_dict_name = "HumanObserved.pkl"
file_path = "HumanObserved-Dataset/HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv"
input_mat = []
n_count = 0
with open(file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
    	input_mat.append(row)
        

author_id_list = []
author_data_dict = {}
for i in input_mat:
	author_id_list.append(i["img_id"][:-1])

author_id_list = set(author_id_list)

for i in input_mat:
    author_id = i["img_id"][:-1]
    sample_id = i["img_id"][-1]
    row = []
    for j in range(1,num_features+1):
        row.append((i["f"+str(j)]))

    try:
        author_data_dict[author_id][sample_id] = row
    #except KeyError as e:
    except:
        author_data_dict[author_id] = {}
        author_data_dict[author_id][sample_id] = row


author_id_list = list(author_data_dict.keys())
print("No. of unique authors: "+ str(len(author_id_list)))
(random.shuffle(author_id_list))

n_train = int(0.8*len(author_id_list))
n_val = int(0.9*len(author_id_list))
author_id_train_list = author_id_list[0:n_train]
author_id_test_list = author_id_list[n_train:n_val]
author_id_val_list = author_id_list[n_val:]
print("No. of train authors: "+ str(len(author_id_train_list)))
print("No. of test authors: "+ str(len(author_id_test_list)))
print("No. of validating authors:"+ str(len(author_id_val_list)))

sample_train_dict = {}
sample_test_dict = {}
sample_val_dict = {}
for author_id in author_id_train_list:
    samples_list = author_data_dict[author_id].keys()
    for s in samples_list:
        sample_train_dict[author_id+s] = author_data_dict[author_id][s]

for author_id in author_id_test_list:
    samples_list = author_data_dict[author_id].keys()
    for s in samples_list:
        sample_test_dict[author_id+s] = author_data_dict[author_id][s]

for author_id in author_id_val_list:
    samples_list = author_data_dict[author_id].keys()
    for s in samples_list:
        sample_val_dict[author_id+s] = author_data_dict[author_id][s]

Y_train = []
Y_test = []
Y_val = []
train_pair_id = [] 
test_pair_id = []
val_pair_id = []
X_train_concat = []
X_test_concat = []
X_val_concat = []

X_train_diff = []
X_test_diff = []
X_val_diff = []

sample_train_id_list = list(sample_train_dict.keys())
print("train id "+str(len(sample_train_id_list)))
sample_test_id_list = list(sample_test_dict.keys())
print("test id "+str(len(sample_test_id_list)))
sample_val_id_list = list(sample_val_dict.keys())
print("val id "+str(len(sample_val_id_list)))
for i in range(len(sample_train_id_list)-1):
    for j in range(i+1,len(sample_train_id_list)):
        train_pair_id.append([sample_train_id_list[i],sample_train_id_list[j]])
        x_row = sample_train_dict[sample_train_id_list[i]] + sample_train_dict[sample_train_id_list[j]]
        x_row = [float(m) for m in x_row] #inline for loop
        X_train_concat.append(x_row)
        x_row = []
        for k in range(len(sample_train_dict[sample_train_id_list[i]])):
            x_row.append(float(sample_train_dict[sample_train_id_list[i]][k]) - float(sample_train_dict[sample_train_id_list[j]][k]))
        X_train_diff.append(x_row)
        if(sample_train_id_list[i][:-1] == sample_train_id_list[j][:-1]):
            Y_train.append(1.0)
        else:
            Y_train.append(0.0)
        # a = str(len(Y_train))
        # print("length Y_train"+a)
Y_train1 = []

Y_train = random.sample(Y_train, len(Y_train))
a = str(len(Y_train))
print("length Y_train"+a)
cte = 0
for i in Y_train:
   
    if cte < 633:
        Y_train1.append(i)
        cte = cte + 1
    else: 
        break

a = str(len(Y_train1))
print("length of Y1 "+a)
X_train_concat = np.array(X_train_concat)
X_train_diff = np.array(X_train_diff)
Y_train = np.array(Y_train)



for i in range(len(sample_test_id_list)-1):
    for j in range(i+1,len(sample_test_id_list)):
        test_pair_id.append([sample_test_id_list[i],sample_test_id_list[j]])
        x_row = sample_test_dict[sample_test_id_list[i]] + sample_test_dict[sample_test_id_list[j]]
        x_row = [float(m) for m in x_row]
        X_test_concat.append(x_row)
        x_row = []
        for k in range(len(sample_test_dict[sample_test_id_list[i]])):
            x_row.append(abs(float(sample_test_dict[sample_test_id_list[i]][k]) - float(sample_test_dict[sample_test_id_list[j]][k])))
        X_test_diff.append(x_row)
        if(sample_test_id_list[i][:-1] == sample_test_id_list[j][:-1]):
            Y_test.append(1.0)
        else:
            Y_test.append(0.0)
Y_test1 = []

Y_test = random.sample(Y_test, len(Y_test))
a = str(len(Y_test))
print("length Y_test"+a)
cte = 0
for i in Y_test:
   
    if cte < 79:
        Y_test1.append(i)
        cte = cte + 1
    else: 
        break

a = str(len(Y_test1))
print("length of Y_test1 "+a)

X_test_concat = np.array(X_test_concat)
X_test_diff = np.array(X_test_diff)
Y_test = np.array(Y_test)




for i in range(len(sample_val_id_list)-1):
    for j in range(i+1,len(sample_val_id_list)):
        val_pair_id.append([sample_val_id_list[i],sample_val_id_list[j]])
        x_row = sample_val_dict[sample_val_id_list[i]] + sample_val_dict[sample_val_id_list[j]]
        x_row = [float(m) for m in x_row]
        X_val_concat.append(x_row)
        x_row = []
        for k in range(len(sample_val_dict[sample_val_id_list[i]])):
            x_row.append(abs(float(sample_val_dict[sample_val_id_list[i]][k]) - float(sample_val_dict[sample_val_id_list[j]][k])))
        X_val_diff.append(x_row)
        if(sample_val_id_list[i][:-1] == sample_val_id_list[j][:-1]):
            Y_val.append(1.0)
        else:
            Y_val.append(0.0)
        Y_val1 = []
        
Y_val1 = []

Y_val = random.sample(Y_val, len(Y_val))
a = str(len(Y_val))
print("length Y_val"+a)
cte = 0
for i in Y_val:
   
    if cte < 79:
        Y_val1.append(i)
        cte = cte + 1
    else: 
        break

a = str(len(Y_val1))
print("length of Y_val1 "+a)

X_val_concat = np.array(X_val_concat)
X_val_diff = np.array(X_val_diff)
Y_val = np.array(Y_val)


print("No. of training sample pairs: " + str(len(Y_train)))
print("No. of test sample pairs: " + str(len(Y_test)))
print("No. of validation sample pairs: " + str(len(Y_val)))



print("Dumping Files (might take some time)")
full_data_dict["X_train_concat"] = X_train_concat
full_data_dict["X_train_diff"] = X_train_diff
full_data_dict["X_val_diff"] = X_val_diff
full_data_dict["X_val_concat"] = X_val_concat
full_data_dict["X_test_diff"] = X_test_diff
full_data_dict["X_test_concat"] = X_test_concat
full_data_dict["Y_train"] = Y_train
full_data_dict["Y_test"] = Y_test
full_data_dict["Y_val"] = Y_val
full_data_dict["train_pair_id"] = train_pair_id
full_data_dict["val_pair_id"] = val_pair_id
full_data_dict["test_pair_id"] = test_pair_id
full_data_dict["author_data_dict"] = author_data_dict
pickle.dump(full_data_dict,open(full_data_dict_name,"wb"))
print("File dumped")



























