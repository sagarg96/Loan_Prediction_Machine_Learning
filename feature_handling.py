import numpy as np
import pandas as pd

# Step 1: Data Preparation
# Loading the training and testing data from numpy files, parsing it as floats and storing it in a pandas dataframe
def parseData():

    #Prepping Training Data
    trainData = np.load('./train.npy')
    df = pd.DataFrame([sub.decode('UTF-8').split(",") for sub in trainData[1:]]) #split rows by commas
    df.columns = trainData[0].decode('UTF-8').split(',') #extract column header names
    df.set_index('id', inplace=True) #set dataframe indices to given indices
    df = df.apply(pd.to_numeric, errors='coerce') #convert data to floats
    df = df.fillna(value=0) # replace all NaN by 0
    df.to_csv("trainData.csv", columns = trainData[0].decode('UTF-8').split(',')) # save as csv File

    #Prepping Testing Data
    testData = np.load('./test.npy')
    tdf = pd.DataFrame([sub.decode('UTF-8').split(",") for sub in testData])
    headers = trainData[0].decode('UTF-8').split(',')
    tdf.columns = headers[:(len(headers)-1)]
    tdf.set_index('id', inplace=True)
    tdf = tdf.apply(pd.to_numeric, errors='coerce')
    tdf = tdf.fillna(value=0)
    tdf.to_csv("testData.csv", columns = trainData[0].decode('UTF-8').split(','))