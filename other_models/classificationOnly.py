import numpy as np
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


""" UNCOMMENT TO INCLUDE READING IN & FORMATTING OF TESTING DATA """
rawTestData = np.load('./ecs171test.npy')
testMatrix = [row.decode('UTF-8').split(',') for row in rawTestData]
del testMatrix[0]
for row in range(0, 55470):
    for col in range(0,770):
        if testMatrix[row][col] == 'NA':
            testMatrix[row][col] = '0'
    testMatrix[row] = np.array(list(map(float, testMatrix[row])))

idVectorTest = np.zeros((55470,), dtype=int)
for row in range(0, 55470):
    idVectorTest[row] = testMatrix[row][0]
testMatrix = np.delete(testMatrix, 0, 1)

# Load raw data for training & testing
rawTrainData = np.load('./ecs171train.npy')

# Parse into lists
trainMatrix = [row.decode('UTF-8').split(',') for row in rawTrainData]
# delete top row ("id", "f1", "f2", ...)
features = trainMatrix[0]
del trainMatrix[0]

# Remove NA values and convert strings to floats
for row in range(0, 50000):
    for col in range(0,771):
        if trainMatrix[row][col] == 'NA':
            trainMatrix[row][col] = '0'
    trainMatrix[row] = np.array(list(map(float, trainMatrix[row])))


# Create ID and Loss vectors by looping through the training data matrix. Also remove
# these columns from training data matrix.
idVectorTrain = np.zeros((50000,), dtype=int)
lossVectorTrain = np.zeros((50000,), dtype=int)

for row in range(0, 50000):
    idVectorTrain[row] = trainMatrix[row][0]
    lossVectorTrain[row] = trainMatrix[row][770]
trainMatrix = np.delete(trainMatrix, 0, 1)
trainMatrix = np.delete(trainMatrix, 769, 1)


# Mutate loss vector for basic classification (1 for default, 0 for no default)
# Also get indexes for loss values for constructing regression matrix later
classif_lossVectorTrain = np.zeros((50000,), dtype=int)
defaultIndexes=[]
for i in range(0, len(lossVectorTrain)):
    if lossVectorTrain[i] > 0:
        classif_lossVectorTrain[i] = 1
        defaultIndexes.append(i)


"""
Comment this section in/out to toggle using the VarianceThreshold test
before using SelectKBest
"""
# print("Variance")
# Reduce feature set by eliminating features with a variance < 20%
sel = VarianceThreshold(threshold=(.98 * (1 - .98)))
trainMatrix1 = sel.fit_transform(trainMatrix)
# defaultTrainData1 = sel.fit_transform(defaultTrainData)
#

"""
This section calls SelectKBest for the classification and regression tests
k = number of features to select
"""
print("SelectKBest")
# Reduce feature set by selecting 50 best features based on f_classif test
k = 50
trainMatrix_new = SelectKBest(f_classif, k).fit_transform(trainMatrix1, classif_lossVectorTrain)
print(trainMatrix_new.shape)
# defaultTrainData_new = SelectKBest(f_regression, k).fit_transform(defaultTrainData, defaultLossData)
# print(defaultTrainData_new.shape)
"""************************************************************************"""

#Train model based on these 100 features to predict default or not
clf = linear_model.SGDClassifier(loss='log', max_iter=800)
scores = cross_val_predict(clf, trainMatrix_new, classif_lossVectorTrain, cv=5)
print("Accuracy of model: %3f" % (accuracy_score(classif_lossVectorTrain, scores)))
print("Accuracy of all 0s: %3f" % (accuracy_score(classif_lossVectorTrain, np.zeros((50000,)))))

print("Calculating MAE")
print("MAE scores vs Binarized: %3f" % (mean_absolute_error(classif_lossVectorTrain, scores)))
print("MAE all 0 vs Binarized: %3f" % (mean_absolute_error(classif_lossVectorTrain, np.zeros((50000,)))))
print("MAE scores vs OG: %3f" % (mean_absolute_error(lossVectorTrain, scores)))
print("MAE all 0 vs OG: %3f" % (mean_absolute_error(lossVectorTrain, np.zeros((50000,)))))

