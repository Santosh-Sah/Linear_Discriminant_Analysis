# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:10:21 2020

@author: Santosh Sah
"""

import pandas as pd
from LinearDiscriminantAnalysisUtils import readLinearDiscriminantAnalysisModel, readLinearDiscriminantAnalysisStandardScaler,readLDA

def predict():
    
    linearDiscriminantAnalysis = readLinearDiscriminantAnalysisModel()
    linearDiscriminantAnalysisStandardScaler = readLinearDiscriminantAnalysisStandardScaler()
    lda = readLDA()

    inputValue = [[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]]
    inputValueDataframe = pd.DataFrame(lda.transform(linearDiscriminantAnalysisStandardScaler.transform(inputValue)))
    
    predictedValue = linearDiscriminantAnalysis.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()