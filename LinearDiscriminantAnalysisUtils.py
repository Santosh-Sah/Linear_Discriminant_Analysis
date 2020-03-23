# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 07:39:47 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importLinearDiscriminantAnalysisDataset(linearDiscriminantAnalysisDatasetFileName):
    
    linearDiscriminantAnalysisDataset = pd.read_csv(linearDiscriminantAnalysisDatasetFileName)
    X = linearDiscriminantAnalysisDataset.iloc[:, 0:13].values
    y = linearDiscriminantAnalysisDataset.iloc[:, 13].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveLinearDiscriminantAnalysisStandardScaler(linearDiscriminantAnalysisStandardScalar):
    
    #Write LinearDiscriminantAnalysisStandardScaler in a picke file
    with open("LinearDiscriminantAnalysisStandardScaler.pkl",'wb') as LinearDiscriminantAnalysisStandardScaler_Pickle:
        pickle.dump(linearDiscriminantAnalysisStandardScalar, LinearDiscriminantAnalysisStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save LinearDiscriminantAnalysisModel as a pickle file.
"""
def saveLinearDiscriminantAnalysisModel(linearDiscriminantAnalysisModel):
    
    #Write LinearDiscriminantAnalysisModel as a picke file
    with open("LinearDiscriminantAnalysisModel.pkl",'wb') as LinearDiscriminantAnalysisModel_Pickle:
        pickle.dump(linearDiscriminantAnalysisModel, LinearDiscriminantAnalysisModel_Pickle, protocol = 2)

"""
read LinearDiscriminantAnalysisStandardScalar from pickel file
"""
def readLinearDiscriminantAnalysisStandardScaler():
    
    #load LinearDiscriminantAnalysisStandardScaler object
    with open("LinearDiscriminantAnalysisStandardScaler.pkl","rb") as LinearDiscriminantAnalysisStandardScaler:
        linearDiscriminantAnalysisStandardScalar = pickle.load(LinearDiscriminantAnalysisStandardScaler)
    
    return linearDiscriminantAnalysisStandardScalar

"""
read LinearDiscriminantAnalysisModel from pickle file
"""
def readLinearDiscriminantAnalysisModel():
    
    #load LinearDiscriminantAnalysisModel model
    with open("LinearDiscriminantAnalysisModel.pkl","rb") as LinearDiscriminantAnalysisModel:
        linearDiscriminantAnalysisModel = pickle.load(LinearDiscriminantAnalysisModel)
    
    return linearDiscriminantAnalysisModel

"""
read X_train from pickle file
"""
def readLinearDiscriminantAnalysisXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readLinearDiscriminantAnalysisXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readLinearDiscriminantAnalysisYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readLinearDiscriminantAnalysisYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveLinearDiscriminantAnalysisYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readLinearDiscriminantAnalysisYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred

def saveTrainingAndTestingDatasetLinearDiscriminantAnalysis(X_train_LinearDiscriminantAnalysis, X_test_LinearDiscriminantAnalysis):
    
    #Write X_train_LinearDiscriminantAnalysis in a picke file
    with open("X_train_LinearDiscriminantAnalysis.pkl",'wb') as X_train_LinearDiscriminantAnalysis_Pickle:
        pickle.dump(X_train_LinearDiscriminantAnalysis, X_train_LinearDiscriminantAnalysis_Pickle, protocol = 2)
    
    #Write X_test_LinearDiscriminantAnalysis in a picke file
    with open("X_test_LinearDiscriminantAnalysis.pkl",'wb') as X_test_LinearDiscriminantAnalysis_Pickle:
        pickle.dump(X_test_LinearDiscriminantAnalysis, X_test_LinearDiscriminantAnalysis_Pickle, protocol = 2)

"""
read X_train_PCA from pickle file
"""
def readLinearDiscriminantAnalysisXTrainLDA():
    
    #load X_train_LDA
    with open("X_train_LinearDiscriminantAnalysis.pkl","rb") as X_train_LDA_pickle:
        X_train_LDA = pickle.load(X_train_LDA_pickle)
    
    return X_train_LDA

"""
read X_test_LDA from pickle file
"""
def readLinearDiscriminantAnalysisXTestLDA():
    
    #load X_test_LDA
    with open("X_test_LinearDiscriminantAnalysis.pkl","rb") as X_test_LDA_pickle:
        X_test_LDA = pickle.load(X_test_LDA_pickle)
    
    return X_test_LDA

def saveLDA(lda):
    
    #Write LDA in a picke file
    with open("LDA.pkl",'wb') as LDA_Pickle:
        pickle.dump(lda, LDA_Pickle, protocol = 2)
        
"""
read LDA from pickle file
"""
def readLDA():
    
    #load LDA
    with open("LDA.pkl","rb") as LDA_pickle:
        lda = pickle.load(LDA_pickle)
    
    return lda
