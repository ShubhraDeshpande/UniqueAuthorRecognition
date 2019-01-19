
# coding: utf-8

# ## Logic Based FizzBuzz Function [Software 1.0]

# In[ ]:


import pandas as pd
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import random
import pickle


print('1 running')



# ## Create Training and Testing Datasets in CSV Format

# In[ ]:


def createInputCSV():
    
    # Why list in Python? 
    # => Mainly because it is an ordered, changable and flexible array. 
    # -> It is easy to be accessed because of the indexing, hence we chose list as our array
    inputData   = []
    outputData  = []
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


    t = []
    n_count = 0
    with open(same_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            t.append(row)
            line_count = line_count + 1
    print ("line count "+ str(line_count))
    with open(diff_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        temp_diff=[]
        for row in csv_reader:
            temp_diff.append(row)
        temp_diff = random.sample(temp_diff, len(temp_diff))
        count = 0

        for row in temp_diff:
            while count < line_count:

                t.append(row)
                count = count + 1

    print ("count of diff = "+ str(count))

    # for i in t:
    #     print(i)
    random.shuffle(t)
    target_vals = []

    for kk in t:
        outputData.append(float(kk["target"]))
    outputData1 = np.array(outputData)

    print("output shaoe",np.shape(outputData1))
    
    
    
    
    
    dataMatrix = [] 
        
    n_count = 0
    pairs=[]
    for i in t:
        pairs.append(i["img_id_A"] + "_" + i["img_id_B"])

    print("length of pairs "+str(len(pairs)))
    n_count = 0
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            dataMatrix.append(row)
            line_count = line_count + 1

    # for i in input_mat:
    #   print(i)
    print ("input size"+str(line_count))

        #listing first row
    img_id_list = []
    author_data_dict = {}
    for i in dataMatrix:
        img_id_list.append(i["img_id"])

    img_id_list = set(img_id_list)

    for i in dataMatrix:
        img_id = i["img_id"]
        row = []
        for j in range(1,num_features+1):
            row.append((i["f"+str(j)]))

        try:
            author_data_dict[img_id] = row
        #except KeyError as e:
        except:
            author_data_dict[img_id] = {}
            author_data_dict[img_id]= row


    # for i in img_id_list:
    #   print(i)
    pair_features_concat = []
    X_train_diff = []
    features_concat = []
    for p in t:
        id1 = p["img_id_A"]
        id2 = p["img_id_B"]
        f_id1 = author_data_dict[id1]
        f_id2 = author_data_dict[id2]
        f_concat = f_id1 + f_id2
        for i in f_concat:
            pair_features_concat.append(float(i))

    for p in t:
        id1 = p["img_id_A"]
        id2 = p["img_id_B"]

        f_id1 = author_data_dict[id1]
        f_id2 = author_data_dict[id2]
        f_diff=[0]*len(f_id1)
        for i in range (num_features):
            f_diff[i] = float(f_id1[i]) - float(f_id2[i])

        inputData.append(f_diff)


    inputData1 = np.array(inputData)

    print("shape",np.shape(inputData1))


    
    train_size = 0.8
    n_train = int(train_size*inputData1.shape[0])
    TrainData = inputData1[0:n_train,:]
    TrainTar = outputData1[0:n_train]
    TestData =  inputData1[n_train:,:]
    TestTar = outputData1[n_train:]
    


    return TrainData,TrainTar,TestData,TestTar
    


# ## Processing Input and Label Data

# In[ ]:


def processData(dataset):
    
    # Why do we have to process?
    # => We have some values of input data and assumed correct answers in the form of labels for those i/ps. 
    # -> We need to process the data to rearrange these values in more usable form
    # -> for eg. in this case, we are converting decimal i/p number into binary and then shifting the number to get every single digit
    # -> of the number as an i/p for one node.
    # -> We process data for converting it into usable form, the form in which we and the model expect our input and the output to be.
    data   = dataset['input1'].values
    labels = dataset['label'].values
    print("shape of data",data.shape)
    
    processedData  = data
    print(processedData)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[ ]:


def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        # =>(1) here, number 10 represents the range i.e. the maximum number of nodes in the input layer. (2) So, as we have processed the data, and converted it from decimal to binary, we get a binary i/p now. We have shifted the data to get
        # -> a single digit of number as a single input node. Now, our training data ranges from 101 to 1000, and testing data ranges from 1 to 100. Hence the maximum
        # -> number of nodes in the i/p layer will be ( 1000(binary) = 1111101000) i.e. len(1111101000)=10; hence, here the range =10.
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)


# In[ ]:


from keras.utils import np_utils

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "1"):
            # Fizzbuzz
            processedLabel.append([1])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)


# ## Model Definition

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import load_model

from keras.utils import to_categorical

import numpy as np

input_size = 9
drop_out = 0.2
first_dense_layer_nodes  = 500
second_dense_layer_nodes = 2

#def get_model():
    
    # Why do we need a model?
    # => Model is nothing but the approach to solve the given problem. set of parameters, instructions, and processes that can, together, find the solution to the given issue.
    # -> Neural network is compared with the brain system. As, the structure of everyone's brain is similar but we still all of us have different abilities to perform different work, 
    # -> in the same way, We define several factors and decide specific architecture that can be the best fit to solve perticular problem.
    # -> hence we need model to serve as a basic structure for performing NN.
    
    # Why use Dense layer and then activation?
    # => Dense layer works as a carrier of weights and the activation function in keras. We can look at it as a link between two layers where the activation function is being applied.
    # -> So we provide the weights through the dense layer and we have an output of input layer, we apply an activation function on them such that the output of the hidden layer will be useful as an input for the next layer in a cinvinient way.
    
    # Why use sequential model with layers?
    # => Sequential models have a property to process the layers sequestially. Though we have a single hidden layer in our model, we do not require the model to share layers in our case, hence we chose sequestial to proceed layer by layer.
    
model = Sequential()

model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
model.add(Activation('relu'))

# Why dropout?
# => In models like this, there are higher chances of overfitting, which can further decrese the accuracy of the model if the range or testing data is changed. 
# -> We drop out some randomly selected inputs or we can say neurons from the training sets to avoid overfitting. In this case, we are dropping 20% neurons.
model.add(Dropout(drop_out))
model.add(Dense(second_dense_layer_nodes))
model.add(Activation('softmax'))

# Why Softmax?1

# => Softmax function converts vectors into probability of output to belong to the perticular class. We have 4 classes and need the probability of every output for being the better fit for one of the classes
model.summary()

# Why use categorical_crossentropy?
# => We use cross entopy to calculate the exact loss that we suffer through and check if the error is not too large. We check the diversion of results from the expected result. 
# -> Used perticularly catagerical cross entropy, because it helps us finding error catagerywise. It works with softmax to chose the wise value of weights to reduce the error in each class.
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#return model
# model.save('saved_model.h5')

#return model


# # <font color='blue'>Creating Training and Testing Datafiles</font>

# In[ ]:


# Create datafiles
#=> inserted data in the function defined above
TrainData, TrainTar,TestData,TestTar = createInputCSV()
TrainTarCat = to_categorical(TrainTar)


validation_data_split = 0.0
num_epochs = 100
model_batch_size = 128
tb_batch_size = 32
early_patience = 100
# a = [0.25,0.50,0.75,0.80,0.99]
# for i in a:
validation_data_split=0.20;
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
history = model.fit(TrainData
                , TrainTarCat
                , validation_split=validation_data_split
                , epochs=num_epochs
                , batch_size=model_batch_size
                , callbacks = [tensorboard_cb,earlystopping_cb]
               )


# # <font color = blue>Training and Validation Graphs</font>

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # <font color = blue>Testing Accuracy [Software 2.0]</font>

# In[ ]:

TestPred = np.argmax(model.predict(TestData),1)
test_acc = np.mean(TestTar == TestPred)
print("test_acc: ", test_acc)


