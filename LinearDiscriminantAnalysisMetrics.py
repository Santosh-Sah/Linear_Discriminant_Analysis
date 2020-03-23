# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:13:43 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from LinearDiscriminantAnalysisUtils import (readLinearDiscriminantAnalysisYTest, readLinearDiscriminantAnalysisYPred)

"""

calculating LinearDiscriminantAnalysis confussion matrix

"""
def testLinearDiscriminantAnalysisConfussionMatrix():
    
    y_test = readLinearDiscriminantAnalysisYTest()
    y_pred = readLinearDiscriminantAnalysisYPred()
    
    linearDiscriminantAnalysisConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(linearDiscriminantAnalysisConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[14  0  0]
    [ 0 16  0]
    [ 0  0  6]]
    
    """
"""
calculating accuracy score

"""

def testLinearDiscriminantAnalysisAccuracy():
    
    y_test = readLinearDiscriminantAnalysisYTest()
    y_pred = readLinearDiscriminantAnalysisYPred()
    
    linearDiscriminantAnalysisConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(linearDiscriminantAnalysisConfussionAccuracy) #1.0%

"""
calculating classification report

"""

def testLinearDiscriminantAnalysisClassificationReport():
    
    y_test = readLinearDiscriminantAnalysisYTest()
    y_pred = readLinearDiscriminantAnalysisYPred()
    
    linearDiscriminantAnalysisConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(linearDiscriminantAnalysisConfussionClassificationReport)
    
    """
              precision    recall  f1-score   support

          1       1.00      1.00      1.00        14
          2       1.00      1.00      1.00        16
          3       1.00      1.00      1.00         6

avg / total       1.00      1.00      1.00        36

    """
    
if __name__ == "__main__":
    #testLinearDiscriminantAnalysisConfussionMatrix()
    #testLinearDiscriminantAnalysisAccuracy()
    testLinearDiscriminantAnalysisClassificationReport()