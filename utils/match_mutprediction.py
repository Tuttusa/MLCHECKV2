import mlCheck
import pandas as pd
import csv as cv
import numpy as np
from utils import util, mlCheck
from operator import add
def func_match_mut_pred(X, model, arr_length):
    obj_mlcheck = mlCheck.runChecker()
    retrain_flag = False
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    mul_cex = paramDict['mul_cex_opt']
    no_of_params = int(paramDict['no_of_params'])
    testIndx = 0
    while testIndx < arr_length:
        temp_store = []
        temp_add_oracle = []
        if not(model.predict(np.reshape(X[testIndx], (1, -1)))[0] ==  model.predict(np.reshape(X[testIndx+1], (1, -1)))[0]):
            retrain_flag = False
            temp_store.append(X[testIndx])
            temp_store.append(X[testIndx+1])
            testIndx += 2
        else:
            retrain_flag = True
            temp_add_oracle.append(X[testIndx])
            temp_add_oracle.append(X[testIndx+1])
            testIndx += 2
        if not retrain_flag:
            if mul_cex == 'True':
                with open('CexSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(temp_store)
            else:
                print('A counter example is found, check it in CexSet.csv file: ', temp_store)
                with open('CexSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(temp_store)
                obj_mlcheck.addModelPred()
                return 1
        else: 
            util.funcAdd2Oracle(temp_add_oracle)
            obj_mlcheck.funcCreateOracle()
    return 0
