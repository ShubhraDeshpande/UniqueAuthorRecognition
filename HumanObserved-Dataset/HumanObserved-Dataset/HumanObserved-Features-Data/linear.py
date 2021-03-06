from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import random
import pickle

maxAcc = 0.0
maxIter = 0
C_Lambda = 0.03
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False
full_data_dict_name = "HumanObserved.pkl"
file_path = "HumanObserved-Dataset/HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv"

diff_file_path = "HumanObserved-Dataset/HumanObserved-Dataset/HumanObserved-Features-Data/diffn_pairs.csv"
same_file_path = "HumanObserved-Dataset/HumanObserved-Dataset/HumanObserved-Features-Data/same_pairs.csv"

print ('ran first')

def GetTargetVector(diff_file_path, same_file_path):
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

    for i in t:
        print(i)

    return t

def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(column) #appending the remaining rows and columns in the dataset
            dataMatrix.append(dataRow)   
    
    
    
    
    
    # for i in input_mat:
    #   print(i)
    

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

    sample_train_dict = {}

    sample_train_id_list = list(author_data_dict.keys())
    train_pair_id= []
    X_train_concat = []
    X_train_diff = []

    for i in range(len(sample_train_id_list)-1):
        for j in range(i+1,len(sample_train_id_list)):
            train_pair_id.append([sample_train_id_list[i],sample_train_id_list[j]])
            #print (train_pair_id[i][j])
            x_row = []
            x_row = author_data_dict[sample_train_id_list[i]] + author_data_dict[sample_train_id_list[j]]
            x_row = [float(m) for m in x_row] #inline for loop
            X_train_concat.append(x_row)

            for k in range(len(author_data_dict[sample_train_id_list[i]])):
                x_row.append(float(author_data_dict[sample_train_id_list[i]][k]) - float(author_data_dict[sample_train_id_list[j]][k]))
            X_train_diff.append(x_row)

    X_train_concat = np.array(X_train_concat)
    X_train_diff = np.array(X_train_diff)

    
    
    
    
    
    
    
    
    
    if IsSynthetic == False :
         #for there values are not significant towards target calculation in our considered case
        dataMatrix = np.transpose(X_train_diff)     
    #print ("Data Matrix Generated..")
    return X_train_diff #generated complete input dataset

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): 
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01))) #0.8 parts of 1 data is used for training purpose hence 80% of both dependant target values and independant input values assiciated with them is taken. 
    t           = rawTraining[:TrainingLen]
    print(t)
#     print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01)) #separating 1% subpart of data for validation
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t


#considering the basis function for phi metrix to be guassian, hence we need sigma (varience) of features. 
#We need varience of vector with itself hence, considering only diagonal matrix and ingoring the rest values in the matrix
#these values will be different for difference clustures but there are higher chances of many of them have at least one complete column zeros. In this case, the determinant will be non zero and hence making it impossible to invert the matrix
#hence we consider overall sigma of all features together, called big sigma
def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic): 
    BigSigma    = np.zeros((len(Data),len(Data))) 
    print(Data) # data is a feature matrix
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        print(vct[0])
        varVect.append(np.var(vct))
    
    for j in range(len(Data)): 
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma #returns covarience matrix refered as Big Sigma
#the output of applying guassian basis function on a sample row with all 41 features, is a scalar value. ref: in report
# hence calculating value of phi_i(x_i)
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L # returns scalar value

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x #gaussian function is e^((-0.5)R^2 / sigma^2) we solve it here with vectorised representation.

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv) 
            #PHI is the design matrix where each row represents guassian function applied on each feature of sample no.1
            #such that row will be [Phi1(x1), phi1(x2), phi1(x3)...]
    #print ("PHI Generated..")
    return PHI #returns design matrix where rows are samples and columns are basis functions.
    # hence the dimentions of PHI is ((80% of Data) X (basis functions))= (58000 X 10)

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0])) 
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda #getting a diagonal matrix with diagonal values matching to lambda
        #In the function of linear regression, we have t = (W)^t . phi(X) hence lambda acts as a regularizing factor
        #We add W = | (inv(((PHI^T)*(PHI))) + (lamba) )* (PHI)^T | * t
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI) 
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W # return W matrix containing associated weights for each basis function. dimentions of W matrix is (10X1)

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI)) #y is the dependatnt target variable, calculated by obtained weights from W matrix
    ##print ("Test Out Generated..")
    return Y #returns PREDICTED target variable for testing data.

def GetErms(VAL_TEST_OUT,ValDataAct): #function check whether the predicted output calculated in function GetValTest is accurate
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2) #We have actual output values of target. We take a difference of actual values and predicted values to know the error in the predictions.
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))  #calculate accuracy in percentage
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    #print(accuracy)
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT)))) 
print('ran')


RawTarget = GetTargetVector(diff_file_path, same_file_path)
RawData   = GenerateRawData(file_path,IsSynthetic)
print('fetched ')


TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape) # dimentions: (55700 X 1)
print(TrainingData.shape) # dimentions : (41 X 55699)

ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget)))) # dimentions: (samples X 1)
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget))) #dimention: (No of features X Samples)
print(ValDataAct.shape)
print(ValData.shape)


TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct)))) #dimentions : (no. of samples X 1)
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct))) #dimentions : (no. of features X no. of samples)
print(ValDataAct.shape)
print(ValData.shape)

ErmsArr = [] #root mean square error, will be used to store erms for different hyperparameter changes
AccuracyArr = [] #accuracy got after tuning several hyper parameter will be stored in this array

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_ #We will be getting 10 mu since number of clustures is 10

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic) #returns big sigma
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent) #generates design matrix
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) #gets weights generated by (PHI(x)^(-1)).t
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) #design matrix for test data
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100) # design matrix for validation data


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)

TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))

W_Now        = np.dot(220, W) #randmly initialised weights
La           = 2 #lamba - used as regularization factor
La_list =[0,1,2] #since the erms of training and testing data is not very dilecting, there are less chances of over and under fitting
learningRate = 0.01 
learningRate_list = [0.01, 0.05,0.25,0.50, 0.70]
L_Erms_Val   = []  # storing L_erms value for different data
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = [] #weight matrix
erms_for_plot =[]


#for j in learningRate_list:
#for j in La_list:
for i in range(0,400):
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i]) #derivative of error function -((t - (W_now)^T).(PHI))
    La_Delta_E_W  = np.dot(La,W_Now) #applying regularization
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W) # adding lambda to the derivative function
    Delta_W       = -np.dot(learningRate,Delta_E) # calculating updatable weights by learning_rate X regularised function
    W_T_Next      = W_Now + Delta_W #actual error in weights and hence, adding error(here error is negative hence it will be substracted), we get new weights
    W_Now         = W_T_Next #assigning new weight to W_Now so that again it could be iterated

    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next)  
    Erms_TR     = GetErms(TR_TEST_OUT,TrainingTarget) #returns erms for training data
    L_Erms_TR.append(float(Erms_TR.split(',')[1])) 

    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val    = GetErms(VAL_TEST_OUT,ValDataAct) #returns validation erms
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))

    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct) #returns testing erms
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
# print('erms, accuracy for training data is:'+str(Erms_TR)+"for La "+str(j))
# print('erms, accuracy for validation data is:'+str(Erms_Val)+"for La "+str(j))
# print('erms, accuracy for testing data is:'+str(Erms_Test)+"for La "+str(j))
iterations_no = [i for i in range(400)]
plt.figure(1)
plt.subplot(311)
plt.plot(iterations_no,L_Erms_TR,'b')
plt.plot()
plt.xlabel('iterations')
plt.ylabel('erms')

plt.subplot(312)
plt.plot(iterations_no,L_Erms_Val,'r')
plt.plot()
plt.xlabel('iterations')
plt.ylabel('erms')

plt.subplot(313)
plt.plot(iterations_no,L_Erms_Test,'g')
plt.plot()
plt.xlabel('iterations')
plt.ylabel('erms')

    print ('----------Gradient Descent Solution--------------------')
    print ("M = 15 \nLambda  = 0.0001 \nlearning_rate ="+j)
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5))) #ERMS of testing must be equal to or slightly greater than testing since in this way we can say that the model is generelised.