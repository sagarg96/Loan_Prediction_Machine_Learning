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

idVectorTest = np.zeros((55470,), dtype=float)
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
idVectorTrain = np.zeros((50000,), dtype=float)
lossVectorTrain = np.zeros((50000,), dtype=int)

for row in range(0, 50000):
    idVectorTrain[row] = trainMatrix[row][0]
    lossVectorTrain[row] = trainMatrix[row][770]
trainMatrix = np.delete(trainMatrix, 0, 1)
trainMatrix = np.delete(trainMatrix, 769, 1)


"""
Comment this section in/out to toggle using the VarianceThreshold test
before using SelectKBest
"""
# print("Variance")
# Reduce feature set by eliminating features with a variance < 20%
sel = VarianceThreshold(threshold=(.95 * (1 - .95)))
trainMatrix1 = sel.fit_transform(trainMatrix)
# defaultTrainData1 = sel.fit_transform(defaultTrainData)
#

"""
This section calls SelectKBest for the classification and regression tests
k = number of features to select
"""
print("SelectKBest")
# Reduce feature set by selecting 50 best features based on f_classif test
k = 100
trainMatrix_new = SelectKBest(f_regression, k).fit_transform(trainMatrix1, lossVectorTrain)
print(trainMatrix_new.shape)
"""************************************************************************"""

#Train model based on these 100 features to predict (for examples that default) the magnitude of a default
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='lad')
scores = cross_val_predict(est, trainMatrix_new, lossVectorTrain, cv=5)

print("Calculating MAE")
print("MAE (Actual loss vs regressionScores): %3f" % (mean_absolute_error(lossVectorTrain, scores)))

np.savetxt("trainScores.csv", scores, delimiter=',')
"""
This section loops through the matrix which only contains the k best features
and identifies them so we can select them for the test data as well
"""
"""CLASSIFICATION """
## SELECTS BEST FEATURES
bestFeatures = []
indexes = []
for newCol in range(0, k):
    for col in range(0, 768):
        for row in range(0, 50000):
            if trainMatrix_new[row][newCol] != trainMatrix[row][col]:
                break;
            if row == 49999 and trainMatrix_new[row][newCol] ==  trainMatrix[row][col]:
                bestFeatures.append(features[col+1])
                indexes.append(col)
print(bestFeatures)

# Construct matching test data matrix
testMatrix_new = np.zeros((len(idVectorTest),k))
for i in range(0, k):
    for row in range(0,len(idVectorTest)):
        testMatrix_new[row][i] = testMatrix[row][indexes[i]]

est.fit(trainMatrix_new, lossVectorTrain)
testScores = est.predict(testMatrix_new)

output = np.zeros((len(testScores),2))
for i in range(0, len(testScores)):
    output[i][0] = idVectorTest[i]
    output[i][1] = testScores[i]

np.savetxt("output.csv", output, delimiter=",")





