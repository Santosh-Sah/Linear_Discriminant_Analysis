# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:14:00 2020

@author: Santosh Sah
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from LinearDiscriminantAnalysisUtils import (saveLinearDiscriminantAnalysisModel, readLinearDiscriminantAnalysisXTrain, readLinearDiscriminantAnalysisYTrain,
                                             saveTrainingAndTestingDatasetLinearDiscriminantAnalysis, readLinearDiscriminantAnalysisXTrainLDA,
                                             readLinearDiscriminantAnalysisXTest, saveLDA)

"""
Train LinearDiscriminantAnalysis model 
"""
def trainLinearDiscriminantAnalysisModel():
    
    X_train = readLinearDiscriminantAnalysisXTrainLDA()
    y_train = readLinearDiscriminantAnalysisYTrain()
        
    linearDiscriminantAnalysis = LogisticRegression(random_state = 1234)
    linearDiscriminantAnalysis.fit(X_train, y_train)
    
    saveLinearDiscriminantAnalysisModel(linearDiscriminantAnalysis)

def selectedFeatureComponentsForModel():
    
    X_train = readLinearDiscriminantAnalysisXTrain()
    X_test = readLinearDiscriminantAnalysisXTest()
    y_train = readLinearDiscriminantAnalysisYTrain()
    
    lda = LDA(n_components = 2)
    lda.fit(X_train, y_train)
    
    X_train = lda.transform(X_train)
    X_test = lda.transform(X_test)
    
    saveLDA(lda)
    saveTrainingAndTestingDatasetLinearDiscriminantAnalysis(X_train, X_test)

if __name__ == "__main__":
    #selectedFeatureComponentsForModel()
    trainLinearDiscriminantAnalysisModel()    
