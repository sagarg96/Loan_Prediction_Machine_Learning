import numpy as np
import pandas as pd
import random
import os
import sys
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


# Step 2: Data Analysis:
# We created a matrix of the correlation of every feature with every other.
# We then took the feature pairs that are highly correlated (correlation between 0.9999 and 1) and returned those feature pairs

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr() # create correlation matrix

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold and corr_matrix.iloc[i, j] < 1: # if value above our threshold
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add( (colname, corr_matrix.columns[j]) ) # add column pairs to set
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return col_corr #set of all the pairs of features with very high correlation

# Step 2: Data Cleaning
# We took the pairs of features obtained from above (about 200 pairs) and created new features from these by taking the difference between 
# the pairs of features and hence obtained 200 new features. This was done in an attempt to solve the issue of overfitting due to the noise in the original features

def makeFeatures(df, flist):
    
    #The features listed below were the ones identified as having the highest correlation after running the above correlation function
    #flist = {('f770', 'f52'), ('f553', 'f532'), ('f483', 'f117'), ('f633', 'f408'), ('f729', 'f644'), ('f722', 'f408'), ('f770', 'f345'), ('f559', 'f246'), ('f729', 'f408'), ('f644', 'f74'), ('f644', 'f354'), ('f417', 'f42'), ('f558', 'f244'), ('f527', 'f274'), ('f608', 'f7'), ('f528', 'f527'), ('f608', 'f478'), ('f559', 'f245'), ('f633', 'f379'), ('f770', 'f42'), ('f504', 'f503'), ('f345', 'f58'), ('f553', 'f8'), ('f569', 'f568'), ('f722', 'f633'), ('f74', 'f36'), ('f354', 'f36'), ('f379', 'f52'), ('f608', 'f488'), ('f767', 'f405'), ('f559', 'f558'), ('f633', 'f427'), ('f494', 'f98'), ('f568', 'f255'), ('f559', 'f247'), ('f770', 'f362'), ('f362', 'f58'), ('f504', 'f106'), ('f633', 'f36'), ('f582', 'f8'), ('f493', 'f98'), ('f484', 'f118'), ('f417', 'f58'), ('f417', 'f36'), ('f577', 'f267'), ('f644', 'f371'), ('f427', 'f58'), ('f371', 'f58'), ('f452', 'f85'), ('f548', 'f234'), ('f770', 'f644'), ('f493', 'f95'), ('f408', 'f74'), ('f408', 'f354'), ('f539', 'f225'), ('f494', 'f96'), ('f729', 'f52'), ('f58', 'f52'), ('f578', 'f577'), ('f484', 'f483'), ('f354', 'f52'), ('f74', 'f52'), ('f379', 'f36'), ('f543', 'f8'), ('f569', 'f255'), ('f453', 'f452'), ('f770', 'f633'), ('f371', 'f52'), ('f770', 'f36'), ('f578', 'f267'), ('f569', 'f257'), ('f722', 'f42'), ('f453', 'f87'), ('f644', 'f417'), ('f345', 'f36'), ('f427', 'f42'), ('f568', 'f257'), ('f417', 'f408'), ('f741', 'f633'), ('f722', 'f36'), ('f493', 'f96'), ('f452', 'f88'), ('f569', 'f254'), ('f345', 'f42'), ('f427', 'f36'), ('f504', 'f108'), ('f362', 'f52'), ('f539', 'f224'), ('f568', 'f254'), ('f483', 'f118'), ('f453', 'f85'), ('f741', 'f644'), ('f578', 'f266'), ('f558', 'f246'), ('f722', 'f644'), ('f644', 'f36'), ('f770', 'f722'), ('f549', 'f548'), ('f528', 'f274'), ('f379', 'f58'), ('f453', 'f88'), ('f768', 'f406'), ('f549', 'f235'), ('f539', 'f227'), ('f452', 'f87'), ('f578', 'f265'), ('f770', 'f417'), ('f549', 'f236'), ('f408', 'f345'), ('f729', 'f58'), ('f494', 'f493'), ('f504', 'f107'), ('f503', 'f108'), ('f379', 'f42'), ('f770', 'f741'), ('f408', 'f52'), ('f532', 'f8'), ('f494', 'f95'), ('f427', 'f408'), ('f503', 'f105'), ('f408', 'f42'), ('f417', 'f52'), ('f484', 'f115'), ('f577', 'f264'), ('f741', 'f42'), ('f608', 'f467'), ('f362', 'f36'), ('f484', 'f116'), ('f538', 'f226'), ('f770', 'f371'), ('f543', 'f532'), ('f408', 'f371'), ('f577', 'f266'), ('f633', 'f417'), ('f770', 'f729'), ('f644', 'f379'), ('f741', 'f52'), ('f408', 'f362'), ('f563', 'f8'), ('f568', 'f256'), ('f484', 'f117'), ('f578', 'f264'), ('f548', 'f237'), ('f644', 'f362'), ('f371', 'f36'), ('f539', 'f538'), ('f608', 'f498'), ('f625', 'f389'), ('f569', 'f256'), ('f74', 'f58'), ('f354', 'f58'), ('f549', 'f237'), ('f453', 'f86'), ('f644', 'f345'), ('f558', 'f247'), ('f741', 'f36'), ('f644', 'f52'), ('f371', 'f42'), ('f563', 'f532'), ('f729', 'f42'), ('f633', 'f371'), ('f722', 'f58'), ('f644', 'f42'), ('f538', 'f225'), ('f770', 'f58'), ('f74', 'f42'), ('f354', 'f42'), ('f427', 'f52'), ('f549', 'f234'), ('f770', 'f408'), ('f494', 'f97'), ('f770', 'f74'), ('f539', 'f226'), ('f770', 'f354'), ('f408', 'f58'), ('f452', 'f86'), ('f483', 'f115'), ('f722', 'f52'), ('f608', 'f439'), ('f573', 'f8'), ('f538', 'f227'), ('f559', 'f244'), ('f608', 'f457'), ('f483', 'f116'), ('f644', 'f58'), ('f362', 'f42'), ('f770', 'f379'), ('f582', 'f532'), ('f644', 'f427'), ('f408', 'f379'), ('f729', 'f633'), ('f503', 'f106'), ('f573', 'f532'), ('f633', 'f345'), ('f558', 'f245'), ('f741', 'f58'), ('f503', 'f107'), ('f729', 'f36'), ('f741', 'f408'), ('f548', 'f236'), ('f633', 'f362'), ('f345', 'f52'), ('f52', 'f42'), ('f548', 'f235'), ('f633', 'f74'), ('f504', 'f105'), ('f633', 'f354'), ('f577', 'f265'), ('f608', 'f508'), ('f644', 'f408'), ('f52', 'f36'), ('f58', 'f36'), ('f493', 'f97'), ('f770', 'f427'), ('f538', 'f224'), ('f42', 'f36')}

    newdf = pd.DataFrame()
    for feature in flist: # for each feature pair in the list
        name = str(feature[0])+"_"+str(feature[1]) 
        newdf[name] = df[feature[0]].sub(df[feature[1]], axis=0) #take the difference of the pair and append it to a new dataframe

    return newdf


def main():

    ''' Step 1: Parsing and Reading data from the csv files we generated in parseData() '''
    # parseData()
    df = pd.read_csv("trainData.csv")
    tdf = pd.read_csv("testData.csv")
    testid = tdf.id
    df.set_index('id', inplace=True)
    tdf.set_index('id', inplace=True)

    df = df.fillna(value=0) #fill nans
    df = df.rename(columns={ df.columns[769]: "loss" })
    tdf = tdf.fillna(value=0) #fill nans

    og_loss = pd.DataFrame(df['loss']) #store loss column


    ''''''''''''' Step 2 & 3: Making New Features ''''''''''''''''''''''''''''''' 
    new_feature_list = correlation(df, 0.9999)
    newFeatures = makeFeatures(traindf, new_feature_list) #training data
    newFeatures_test = makeFeatures(testdf, new_feature_list) #testing data
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    #Binarizing the loss in the training data to run it through our classfication model first
    traindf = pd.DataFrame(df)
    traindf['loss'][traindf['loss'] > 0] = 1
    loss = traindf['loss']
    traindf = traindf.drop(['loss'], axis =1) #remove loss column for training
    testdf = pd.DataFrame(tdf)


    ''''''''''' Step 4: Applying PCA to reduce Dimentionality '''''''''''''''''
    # We initially ran the model using all features and then using the SelectKbest function in scklearn however, both methods led to overfitting of our model.
    # Hence we decided to use Principal Component Analysis to reduce the dimentionality of our features 

    traindf = StandardScaler().fit_transform(traindf) # Normalize the data
    pca = PCA(n_components=35) # Apply PCA to reduce the original 769 vectors to 35 components
    '''*Note: We chose the number 35 after trial and error as it seemed to consistly provide the best results for our model'''
    principalComponents = pca.fit_transform(traindf) # Fit the data based on the PCA model

    # Same as above for testing data
    testdf = StandardScaler().fit_transform(testdf)
    pca_test = PCA(n_components=35)
    principalComponentsTest = pca_test.fit_transform(testdf)

    principalDf1 = pd.DataFrame(data = principalComponents) #PCA data frame for training data
    principalDf1_test = pd.DataFrame(data = principalComponentsTest) #PCA data frame for testing data


    ''''''''''''''''' Step 5: Data Manipulation '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Finally we took the 35 PCA components we obtained earlier and combined them with the features we had created in Step 2 based on correlations
    # and used this final feature set of roughly 240 features in order to train our model (@Janet explain why here)
    newFeatures.reset_index(drop=True, inplace=True)
    principalDf = pd.concat([principalDf1, newFeatures], axis=1) # merge the 2 data frames for training data

    newFeatures_test.reset_index(drop=True, inplace=True)
    principalDf_test = pd.concat([principalDf1_test, newFeatures_test], axis=1) #testing data


    ''''''''''''''''''''' Step 6: Binary Classification: Using Logistic Regression with SGD ''''''''''''''''''''''''''''
    # We first use binary classification to predict whether or not a loan will default (i.e loss is 0 or 1 here)

    clf = linear_model.SGDClassifier(loss='log', max_iter=800) #define classification model
    scores = cross_val_predict(clf, principalDf, loss, cv=5) #do a 5 fold cross validation on th data set using the model
    print("Accuracy of model: %3f" % (accuracy_score(loss, scores))) #Checking accuracies
    print("Accuracy of all 0s: %3f" % (accuracy_score(loss, np.zeros((50000,)))))

    clf.fit(principalDf, loss) #fitting our data with respect to the model
    prediction = clf.predict(principalDf) #storing the model predictions ( for training )
    prediction_test = clf.predict(principalDf_test) #storing the model predictions ( for testing )

    ''''''''''''''''''''' Step 7: Apply regression to obtain the severity of loss ''''''''''''''''''''''''''''
    # Now for all those loan samples above where we predicted that the loan will default, we use a gradient boosted regression tree model
    # to classify the severity of the loss based on the original loss values

    # create loss training vector
    lossTrainingIndex = []
    for i in range(0, len(loss)):
        if loss.as_matrix()[i] == 1:
            lossTrainingIndex.append(i)

    regressionDF = principalDf.loc[lossTrainingIndex[:]]
    regressionLoss = pd.DataFrame(og_loss)
    regressionLoss = regressionLoss[regressionLoss['loss'] > 0] # keep only rows with loss > 0 in regression dataframe

    a = regressionLoss.as_matrix()
    a = a.flatten()

    # GBT Model with learning rate of 0.1 
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='lad')
    regressionScores = cross_val_predict(est, regressionDF, a, cv=3) # 3 fold cross validation
   
    est.fit(regressionDF, a) #fit the data based on the model

    #keeping only required rows
    lossPredictedIndex = []
    for i in range(0, len(prediction)):
        if prediction[i] == 1:
            lossPredictedIndex.append(i)

    toPredict_reg = principalDf.loc[lossPredictedIndex[:]]
    prediction_reg = est.predict(toPredict_reg)

    #assign values back to original indices
    for i in range(0, len(prediction_reg)):
         prediction[lossPredictedIndex[i]] = prediction_reg[i]

    # print("Final MAE (prediction vs og_loss): %3f" % (mean_absolute_error(og_loss, prediction)))
    # print("Control MAE (all-0s vs og_loss): %3f" % (mean_absolute_error(og_loss, np.zeros((50000,)))))

    #Same for testing data
    lossPredictedIndex_test = []
    for i in range(0, len(prediction_test)):
        if prediction_test[i] ==1:
            lossPredictedIndex_test.append(i)

    toPredict_reg_test = principalDf_test.loc[lossPredictedIndex_test[:]]
    prediction_reg_test = est.predict(toPredict_reg_test) #Finally making the predictions based on the regression model

    ''''''''' Finally a script to generate the id and predicted loss data and save the file '''''''''
    for i in range(0, len(prediction_reg_test)):
         prediction_test[lossPredictedIndex_test[i]] = prediction_reg_test[i]

    ids = np.array(testid.tolist())
    output_tuples = np.column_stack((ids, prediction_test))
    output_tuples = output_tuples.astype(str)
    output_tuples = np.insert(output_tuples, 0, ['id', 'loss'], axis=0 )
    np.savetxt("output_tuples.csv", output_tuples, delimiter=",", fmt="%s")

if __name__ == "__main__":
    main()