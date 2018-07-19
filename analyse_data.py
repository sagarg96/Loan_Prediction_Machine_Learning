import pandas as pd 
import numpy as np

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