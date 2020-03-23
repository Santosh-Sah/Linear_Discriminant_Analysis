# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:04:14 2020

@author: Santosh Sah
"""

from sklearn.preprocessing import StandardScaler
from LinearDiscriminantAnalysisUtils import (importLinearDiscriminantAnalysisDataset, saveTrainingAndTestingDataset, saveLinearDiscriminantAnalysisStandardScaler)

def preprocess():
    
    X_train, X_test, y_train, y_test = importLinearDiscriminantAnalysisDataset("Linear_Discriminant_Analysis_Wines.csv")
    
    linearDiscriminantAnalysisStandardScalar = StandardScaler()
    
    linearDiscriminantAnalysisStandardScalar.fit(X_train)
    saveLinearDiscriminantAnalysisStandardScaler(linearDiscriminantAnalysisStandardScalar)
    
    X_train = linearDiscriminantAnalysisStandardScalar.transform(X_train)
    X_test = linearDiscriminantAnalysisStandardScalar.transform(X_test)
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()