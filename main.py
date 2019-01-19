import os

print("linear regression start")
cmd = "python linear_regression.py"
os.system(cmd)
print("linear regression done")
print("data processing start")
cmd = "python data_processing.py"
os.system(cmd)
print("data processing done")
cmd = "python logistic_regression.py"
os.system(cmd)
print("logistic regression done")
cmd = "python neuralNetwork.py"
os.system(cmd)
print("neural Network done")