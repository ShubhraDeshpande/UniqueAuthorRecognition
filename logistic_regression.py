from numpy import genfromtxt
import csv
import numpy as np 
import pickle
import random
import pickle
import matplotlib.pyplot as plt
a = input("select dataset 1. for HumanObserved and 2 for GSC Dataset ")
if( a == "1"):
    num_features = 9
    file_path ="HumanObserved-Dataset/HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv"
    diff_file_path ="HumanObserved-Dataset/HumanObserved-Dataset/HumanObserved-Features-Data/diffn_pairs.csv"
    same_file_path ="HumanObserved-Dataset/HumanObserved-Dataset/HumanObserved-Features-Data/same_pairs.csv"
    a = "HumanObserved"

if(a=="2"):
    num_features = 512
    file_path = "GSC-Dataset/GSC-Dataset/GSC-Features-Data/GSC-Features.csv"
    diff_file_path = "GSC-Dataset/GSC-Dataset/GSC-Features-Data/diffn_pairs.csv"
    same_file_path = "GSC-Dataset/GSC-Dataset/GSC-Features-Data/same_pairs.csv"
    a =" GSC-Dataset"
print(str(a))
print("number of features "+str(num_features))

def sigmoid(X,w):
    X = X.T 
    temp =np.zeros((len(w), len(X[0])))
    temp = np.dot(w,X)
    t = np.zeros((len(w), len(X[0])))
    denom = np.ones(temp.shape)
    temp = (-1) * temp
    denom = denom + np.exp(temp)
    t = 1/denom 
    return t
    #return (1 / (1 + np.exp(-x)))

def cost_function(features, labels, weights):
    
    observations = len(labels)
    
    predictions = sigmoid(features, weights)
    class1_cost = []
    a = np.log(predictions)
    a = np.array([a])
    class1_cost = (-1)* (  labels* a )
    class2_cost = (1-labels) * a 
    cost = class1_cost - class2_cost
    cost = cost.sum()/observations

    return cost

def gradient_descent(features, labels, weights, lr, iterations):
    
    N = len(features)

    predictions = sigmoid(features, weights)
    # predictions = np.array(predictions)
    # weights = np.array(weights)

    print("shape pred",predictions.shape)
    print("shape lebel",labels.shape)
    print("shape feature",features.shape)
    for iteration in range(iterations):

        
        gradient = np.dot( features.T, (predictions - labels.T))
        #print("shape gradient",gradient.shape)
        gradient /= N
        gradient *= lr
        weights = weights - gradient

    return weights






full_data_dict_name = "HumanObserved.pkl"
full_data_dict = pickle.load(open(full_data_dict_name,"rb"))

X_train_concat = full_data_dict["X_train_concat"]
X_train_diff = full_data_dict["X_train_diff"]
X_test_concat = full_data_dict["X_test_concat"]
X_test_diff = full_data_dict["X_test_diff"]
Y_train = full_data_dict["Y_train"]
Y_test = full_data_dict["Y_test"]


print("Starting for concatenated features")
print("Training Model")
X_train = X_train_concat
B = np.array([0]*(X_train.shape[1]+1))
x0 = np.ones(X_train.shape[0])
X = np.c_[x0, X_train]
Y = Y_train
inital_cost = cost_function(X, Y, B)
print("Initial Cost: "+ str(inital_cost))
alpha = 0.001
n_epochs = 2000
newB = gradient_descent(X, Y, B, alpha, n_epochs)
Y_pred = X.dot(newB)

print("\n")
print("Testing Model")
X_test = X_test_concat
x0 = np.ones(X_test.shape[0])
X = np.c_[x0, X_test]
Y = Y_test
test_cost = cost_function(X, Y, newB)
print("Test Cost: "+ str(test_cost))
Y_pred = sigmoid(X,newB)
test_pred = Y_pred > 0.4
acc = np.mean(Y == test_pred)
print("accuracy: ", acc)
print("Conatenated features done")


print("\n \n")

print("Starting for Diff features")
print("Training Model")
X_train = X_train_diff
B = np.array([0]*(X_train.shape[1]+1))
x0 = np.ones(X_train.shape[0])
X = np.c_[x0, X_train]
Y = Y_train
inital_cost = cost_function(X, Y, B)
print("Initial Cost: "+ str(inital_cost))
alpha = 0.001
n_epochs = 2000 
newB = gradient_descent(X, Y, B, alpha, n_epochs)

Y_pred = X.dot(newB)


print("\n")
print("Testing Model")
X_test = X_test_diff
x0 = np.ones(X_test.shape[0])
X = np.c_[x0, X_test]
Y = Y_test
test_cost = cost_function(X, Y, newB)
print("Test Cost: "+ str(test_cost))
Y_pred = sigmoid(X,newB)
test_pred = Y_pred > 0.45
acc = np.mean(Y == test_pred)
print("accuracy: ", acc)
print("Diff features done")

