# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:58:49 2020

@author: Santosh Sah
"""

from LinearDiscriminantAnalysisUtils import (readLinearDiscriminantAnalysisXTestLDA, readLinearDiscriminantAnalysisModel,
                                     saveLinearDiscriminantAnalysisYPred)

"""
test the model on testing dataset
"""
def testLogisticRegressionModel():
    
    X_test = readLinearDiscriminantAnalysisXTestLDA()
    
    linearDiscriminantAnalysisModel = readLinearDiscriminantAnalysisModel()
    
    y_pred = linearDiscriminantAnalysisModel.predict(X_test)
    saveLinearDiscriminantAnalysisYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testLogisticRegressionModel()