#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import csv as cv
from utils import Pruning
import numpy as np


# In[1]:


def funcAddCex2CandidateSet():
    df = pd.read_csv('TestDataSMT.csv')
    data = df.values
    with open('CandidateSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
        
        
def funcAddCexPruneCandidateSet(tree_model):
    df = pd.read_csv('TestDataSMT.csv')
    data = df.values
    
    with open('TestDataSMTMain.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
    
    df = pd.read_csv('OracleData.csv')
    #Pruning by negating the data instance
    Pruning.funcPrunInst(df, False)
    dfInst = pd.read_csv('CandidateSetInst.csv')
    dataInst = dfInst.values
    with open('CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)
    

    #Pruning by toggling the branch conditions
    Pruning.funcPrunBranch(df, tree_model)
    dfBranch = pd.read_csv('CandidateSetBranch.csv')
    dataBranch = dfBranch.values    
    with open('CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataBranch)  


def funcCheckDuplicate(pairfirst, pairsecond, testMatrix):
    pairfirstList = pairfirst.tolist()
    pairsecondList = pairsecond.tolist()
    testDataList = testMatrix.tolist()
    
    for i in range(0, len(testDataList)-1):
        if(pairfirstList == testDataList[i]):
            if(pairsecondList == testDataList[i+1]):
                return True
            #elif(pairsecondList == testDataList[i-1]):
             #   return True
    
    dfTest = pd.read_csv('TestSet.csv')
    dataTest = dfTest.values
    dataTestList = dataTest.tolist()
    for i in range(0, len(dataTestList)-1):
        if(pairfirstList == dataTestList[i]):
            if(pairsecondList == dataTestList[i+1]):
                return True
    return False


def funcCheckTriplicate(pairfirst, pairsecond, pairthird, testMatrix):
    pairfirstList = pairfirst.tolist()
    pairsecondList = pairsecond.tolist()
    pairthirdList = pairthird.tolist()
    testDataList = testMatrix.tolist()

    for i in range(0, len(testDataList) - 2):
        if pairfirstList == testDataList[i]:
            if pairsecondList == testDataList[i + 1]:
                if pairthirdList == testDataList[i+2]:
                    return True

    dfTest = pd.read_csv('TestSet.csv')
    dataTest = dfTest.values
    dataTestList = dataTest.tolist()
    for i in range(0, len(dataTestList) - 2):
        if pairfirstList == dataTestList[i]:
            if pairsecondList == dataTestList[i + 1]:
                if pairthirdList == dataTestList[i+2]:
                    return True
    return False



def funcCheckCex():
    dfCandidate = pd.read_csv('CandidateSet.csv')
    with open('Cand-set.csv', 'w', newline='') as csvfile:
        fieldnames = dfCandidate.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)

    dataCandidate = dfCandidate.values
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    no_of_param = int(paramDict['no_of_params'])
    
    candIndx = 0
    testIndx = 0
    
    while candIndx <= dfCandidate.shape[0]-1:
        if no_of_param == 3:
            dfTest_set = pd.read_csv('TestSet.csv')
            dataTest_set = dfTest_set.values
            pairfirst = dataCandidate[candIndx]
            pairsecond = dataCandidate[candIndx + 1]
            pairthird = dataCandidate[candIndx + 2]
            #if funcCheckTriplicate(pairfirst, pairsecond, pairthird, dataTest_set):
            candIndx = candIndx + 3
            '''
            else:
                with open('TestSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerow(pairfirst)
                    writer.writerow(pairsecond)
                with open('Cand-set.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerow(pairfirst)
                    writer.writerow(pairsecond)
                candIndx = candIndx + 3
            '''
        elif no_of_param == 2:
            dfTest_set = pd.read_csv('TestSet.csv')
            dataTest_set = dfTest_set.values
            pairfirst = dataCandidate[candIndx]
            pairsecond = dataCandidate[candIndx+1]
            if funcCheckDuplicate(pairfirst, pairsecond, dataTest_set):
                candIndx = candIndx+2
            else:
                with open('TestSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerow(pairfirst)
                    writer.writerow(pairsecond)
                with open('Cand-set.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerow(pairfirst)
                    writer.writerow(pairsecond)
                candIndx = candIndx + 2
        elif no_of_param == 1:
            dfTest_set = pd.read_csv('TestSet.csv')
            dataTest_set = dfTest_set.values
            if not np.round(dataCandidate[candIndx].tolist(), 10) in np.round(dataTest_set.tolist(), 10):
                with open('TestSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerow(dataCandidate[candIndx])
                with open('Cand-set.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerow(dataCandidate[candIndx])
                candIndx = candIndx + 1
            else:
                candIndx = candIndx + 1
            #print(dfSet))
            #dfSet = pd.read_csv('TestSet.csv')
            #print(dfSet)

    '''    
    #Eliminating the rows with zero values    
    
    dfTest = dfTest[(dfTest.T != 0).any()]
    dfTest.to_csv('TestSet.csv', index = False, header = True)  
    
    #Eliminating the rows with zero values    
    dfCand = pd.read_csv('Cand-set.csv')
    dfCand = dfCand[(dfCand.T != 0).any()]
    dfCand.to_csv('Cand-set.csv', index = False, header = True)
    '''

def funcAddCexPruneCandidateSet4DNN():
    df = pd.read_csv('TestDataSMT.csv')
    data = df.values
    
    with open('TestDataSMTMain.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
    
    df = pd.read_csv('OracleData.csv')
    #Pruning by negating the data instance
    Pruning.funcPrunInst(df, True)
    dfInst = pd.read_csv('CandidateSetInst.csv')
    dataInst = dfInst.values
    with open('CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)
    
        
