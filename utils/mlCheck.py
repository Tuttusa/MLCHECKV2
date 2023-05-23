#!/usr/bin/env python



import pandas as pd
import csv as cv
import numpy as np
import random as rd
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
from itertools import groupby
import re
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, time
from utils import tree2Logic, Pruning, ReadZ3Output, processCandCex, util, assume2logic, assert2logic
from utils import trainDNN, DNN2logic
from joblib import dump, load
from multi_utils import multiLabelMain
from utils.PytorchDNNStruct import Net
import time
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from utils import logic2assert







class generateData:
    def __init__(self, feNameArr, feTypeArr, minValArr, maxValArr):
        self.nameArr = feNameArr
        self.typeArr = feTypeArr
        self.minArr = minValArr
        self.maxArr = maxValArr
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)


    # function to search for duplicate test data
    def binSearch(self, alist, item):
        if len(alist) == 0:
            return False
        else:
            midpoint = len(alist) // 2
            if alist[midpoint] == item:
                return True
            else:
                if item < alist[midpoint]:
                    return self.binSearch(alist[:midpoint], item)
                else:
                    return self.binSearch(alist[midpoint + 1:], item)

    # Function to generate a new sample
    def funcGenData(self):
        tempData = np.zeros((1, len(self.nameArr)), dtype=object)

        #Change VM--- without any loop numpy random
        for k in range(0, len(self.nameArr)):

            fe_type = self.typeArr[k]
            if 'int' in fe_type:

                tempData[0][k] = rd.randint(self.minArr[k], self.maxArr[k])
            else:
                tempData[0][k] = rd.uniform(0, self.maxArr[k])
        return tempData

    # Function to check whether a newly generated sample already exists in the list of samples
    def funcCheckUniq(self, matrix, row):
        row_temp = row.tolist()
        matrix_new = matrix.tolist()
        if row_temp in matrix_new:
            return True
        else:
            return False

    # Function to combine several steps
    def funcGenerateTestData(self):
        tst_pm = int(self.paramDict['no_of_train'])
        testMatrix = np.zeros(((tst_pm + 1), len(self.nameArr)), dtype=object)
        i = 0
        while i <= tst_pm:
            # Generating a test sample
            temp = self.funcGenData()
            # Checking whether that sample already in the test dataset
            flg = self.funcCheckUniq(testMatrix, temp)
            if not flg:
                for j in range(0, len(self.nameArr)):
                    testMatrix[i][j] = temp[0][j]
                i = i + 1

        with open('TestingData.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(self.nameArr)
            writer.writerows(testMatrix)

        if self.paramDict['train_data_available'] == 'True':
            dfTrainData = pd.read_csv(self.paramDict['train_data_loc'])
            self.generateTestTrain(dfTrainData, int(self.paramDict['train_ratio']))
        with open('TestSet.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(self.nameArr)
        with open('CexSet.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(self.nameArr)

    # Function to take train data as test data
    def generateTestTrain(self, dfTrainData, train_ratio):
        print(train_ratio)
        tst_pm = round((train_ratio * dfTrainData.shape[0])/100)
        data = dfTrainData.values
        testMatrix = np.zeros(((tst_pm + 1), dfTrainData.shape[1]))
        testCount = 0
        ratioTrack = []
        noOfRows = dfTrainData.shape[0]
        while testCount <= tst_pm:
            ratio = rd.randint(0, noOfRows - 1)
            if testCount >= 1:
                flg = self.binSearch(ratioTrack, ratio)
                if not flg:
                    ratioTrack.append(ratio)
                    testMatrix[testCount] = data[ratio]
                    testCount = testCount + 1
            if testCount == 0:
                ratioTrack.append(ratio)
                testMatrix[testCount] = data[ratio]
                testCount = testCount + 1
        with open('TestingData.csv', 'a', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerows(testMatrix)


class dataFrameCreate(NodeVisitor):
    def __init__(self):
        self.feName = None
        self.feType = None
        self.feMinVal = -99999
        self.feMaxVal = 0

    def generic_visit(self, node, children):
        pass

    def visit_feName(self, node, children):
        self.feName = node.text

    def visit_feType(self, node, children):
        self.feType = node.text

    def visit_minimum(self, node, children):
        digit = float(re.search(r'[+-]?([0-9]*[.])?[0-9]+', node.text).group(0))
        self.feMinVal = digit

    def visit_maximum(self, node, children):
        digit = float(re.search(r'[+-]?([0-9]*[.])?[0-9]+', node.text).group(0))
        self.feMaxVal = digit


class readXmlFile:

    def __init__(self, fileName):
        self.fileName = fileName

    def funcReadXml(self):
        grammar = Grammar(
            r"""

            expr             = name / type / minimum / maximum / xmlStartDoc / xmlStartInps / xmlEndInps / xmlStartInp /
                                                                        xmlEndInp / xmlStartOut / xmlEndOut
            name             = xmlStartNameTag feName xmlEndNameTag
            type             = xmlStartTypeTag feType xmlEndTypeTag
            minimum          = xmlStartMinTag number xmlEndMinTag
            maximum          = xmlStartMaxTag number xmlEndMaxTag
            xmlStartDoc      = '<Schema>'
            xmlStartInps     = "<input>"
            xmlEndInps       = "</input>"
            xmlStartOut      = "<output>"
            xmlEndOut      = "</output>"
            xmlStartInp      = "<feature>"
            xmlEndInp        = "</feature>"
            xmlStartNameTag  = "<name>"
            xmlEndNameTag    = "</name>"
            xmlStartTypeTag  = "<type>"
            xmlEndTypeTag    = "</type>"
           
            xmlStartMinTag   = "<minVal>"
            xmlEndMinTag     = "</minVal>"
            xmlStartMaxTag   = "<maxVal>"
            xmlEndMaxTag     = "</maxVal>"
            feName           = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
            feType           = ~"[A-Z 0-9]*"i
            number           = ~"[+-]?([0-9]*[.])?[0-9]+"
            """
        )

        with open(self.fileName) as f1:
            file_content = f1.readlines()
        file_content = [x.strip() for x in file_content]
        feNameArr = []
        feTypeArr = []
        minValArr = []
        maxValArr = []
        feName_type = {}
        feMinVal = {}
        feMaxVal = {}
        fe_type = ''
        for lines in file_content:
            tree = grammar.parse(lines)
            dfObj = dataFrameCreate()
            dfObj.visit(tree)

            if dfObj.feName is not None:
                feNameArr.append(dfObj.feName)
                fe_name = dfObj.feName
            if dfObj.feType is not None:
                feTypeArr.append(dfObj.feType)
                fe_type = dfObj.feType
                feName_type[fe_name] = fe_type
            if dfObj.feMinVal != -99999:
                if 'int' in fe_type:
                    minValArr.append(int(dfObj.feMinVal))
                    feMinVal[fe_name] = int(dfObj.feMinVal)
                else:
                    minValArr.append(dfObj.feMinVal)
                    feMinVal[fe_name] = float(dfObj.feMinVal)
            if dfObj.feMaxVal != 0:
                if 'int' in fe_type:
                    maxValArr.append(int(dfObj.feMaxVal))
                    feMaxVal[fe_name] = int(dfObj.feMaxVal)
                else:
                    maxValArr.append(dfObj.feMaxVal)
                    feMaxVal[fe_name] = float(dfObj.feMaxVal)
        try:
            with open('feNameType.csv', 'w', newline='') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in feName_type.items():
                    writer.writerow([key, value])
            with open('feMinValue.csv', 'w', newline='') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in feMinVal.items():
                    writer.writerow([key, value])
            with open('feMaxValue.csv', 'w', newline='') as csv_file:
                writer = cv.writer(csv_file)
                for key, value in feMaxVal.items():
                    writer.writerow([key, value])
        except IOError:
            print("I/O error")

        with open('param_list.csv') as csv_file:
            reader = cv.reader(csv_file)
            paramList= dict(reader)

        final_dataset = np.zeros((len(list(paramList)) * 5, len(list(feName_type))))


        with open('FeatureValueRange.csv', 'w', newline='') as csv_file:
            writer = cv.writer(csv_file)
            writer.writerow(list(feName_type))
            writer.writerow(minValArr)
            writer.writerow(maxValArr)

        genDataObj = generateData(feNameArr, feTypeArr, minValArr, maxValArr)
        genDataObj.funcGenerateTestData()


class makeOracleData:

    def __init__(self, model):
        self.model = model
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

    def funcGenOracle(self):
        dfTest = pd.read_csv('TestingData.csv')
        dataTest = dfTest.values
        predict_list = np.zeros((1, dfTest.shape[0]))
        X = dataTest[:, :-1]

        if 'numpy.ndarray' in str(type(self.model)):
            for i in range(0, X.shape[0]):
                predict_list[0][i] = np.sign(np.dot(self.model, X[i]))
                dfTest.loc[i, 'Class'] = int(predict_list[0][i])

        else:
            if self.paramDict['model_type'] == 'Pytorch':
                X = torch.tensor(X, dtype=torch.float32)
                predict_class = []
                for i in range(0, X.shape[0]):
                    predict_prob = self.model(X[i].view(-1, X.shape[1]))
                    predict_class.append(int(torch.argmax(predict_prob)))
                for i in range(0, X.shape[0]):
                    dfTest.loc[i, 'Class'] = predict_class[i]
            else:
                predict_class = self.model.predict(X)
                for i in range(0, X.shape[0]):
                    if self.paramDict['regression'] == 'yes':
                        dfTest.loc[i, 'Class'] = predict_class[i]
                    else:
                        dfTest.loc[i, 'Class'] = int(predict_class[i])

        dfTest.to_csv('OracleData.csv', index=False, header=True)


class propCheck:
    
    def __init__(self, max_samples=None, deadline=None, model=None, xml_file='', mul_cex=False,
                white_box_model=None, no_of_layers=None, layer_size=None, no_of_class=None,
                no_EPOCHS=None, train_data_available=False, train_data_loc='', multi_label=False,
                model_type=None, model_path='', no_of_train=None, train_ratio=None,
                regression = 'no', bound_cex = False, upper_bound=None, lower_bound=None, instance_list = [],
                nn_library='', bound_list=[], bound_all_features = False, solver=None):
        
        self.paramDict = {}

        param_list_dict = {}
        keys = [x for x in range(0, len(instance_list))]
        for el in keys:
            param_list_dict[keys[el]] = instance_list[el]
        with open('param_list.csv', 'w', newline='') as csv_file:
            writer = cv.writer(csv_file)
            for key, value in param_list_dict.items():
                writer.writerow([value, key])
                
        no_of_params = len(instance_list)

        if white_box_model == 'DNN' or multi_label:
            if no_of_class is None:
                raise Exception('Please provide the number of classes the dataset contain')
            else:
                self.paramDict['no_of_class'] = no_of_class

        self.paramDict['bound_all_features'] = bound_all_features
        self.paramDict['solver'] = solver
        if nn_library == '':
            self.paramDict['nn-library'] = 'sklearn'
        else:
            self.paramDict['nn-library'] = nn_library
        self.paramDict['bound_list'] = bound_list
        if multi_label:
            multiLabelMain.multiLabelPropCheck(no_of_params=no_of_params, max_samples=max_samples, deadline=deadline, model=model,
                                               xml_file=xml_file, no_of_class=no_of_class, mul_cex=mul_cex,
                                               white_box_model=white_box_model, no_of_layers=no_of_layers,
                                               layer_size=layer_size, no_EPOCHS=no_EPOCHS, model_path=model_path, no_of_train=None,
                                               train_ratio=None, model_type=model_type)
        else:
            if max_samples is None:
                self.max_samples = 1000
            else:
                self.max_samples = max_samples
            self.paramDict['max_samples'] = self.max_samples
            self.paramDict['regression'] = regression
            if bound_cex == True:
                self.paramDict['bound_cex'] = bound_cex
                self.paramDict['upper_bound'] = upper_bound
                self.paramDict['lower_bound'] = lower_bound
            else:
                self.paramDict['bound_cex'] = bound_cex
        
            if deadline is None:
                self.deadline = 1000
            else:
                self.deadline = deadline
            self.paramDict['deadlines'] = self.deadline
        
            if white_box_model is None:
                self.white_box_model = 'Decision tree'
            else:
                self.white_box_model = white_box_model
            self.paramDict['white_box_model'] = self.white_box_model    

            if self.white_box_model == 'DNN':
                if (no_of_layers is None) and (layer_size is None):
                    self.no_of_layers = 2
                    self.layer_size = [10]
                elif no_of_layers is None:
                    self.no_of_layers = 2
                    self.layer_size = layer_size
                elif layer_size is None:
                    self.no_of_layers = no_of_layers
                    self.layer_size = [10]
                elif (len(layer_size) > 100) or(no_of_layers > 5):
                    raise Exception("White-box model is too big to translate")
                    sys.exit(1)    
                else:
                    self.no_of_layers = no_of_layers
                    self.layer_size = layer_size
                self.paramDict['no_of_layers'] = self.no_of_layers
                self.paramDict['layer_size'] = self.layer_size 
           
            if no_EPOCHS is None:
                self.paramDict['no_EPOCHS'] = 20
            else:
                self.paramDict['no_EPOCHS'] = no_EPOCHS
            
            if (no_of_params is None) or (no_of_params > 3):
                raise Exception("Please provide a value for no_of_params or the value of it is too big")
            else:
                self.no_of_params = no_of_params
            self.paramDict['no_of_params'] = self.no_of_params   
            self.paramDict['mul_cex_opt'] = mul_cex
            self.paramDict['multi_label'] = False

            if xml_file == '':
                raise Exception("Please provide a file name")
            else:
                try:
                    self.xml_file = xml_file
                    self.paramDict['xml_file'] = xml_file
                except Exception as e:

                    raise Exception("File does not exist")

            if model_type == 'sklearn':
                if model is None:
                    if model_path == '':
                        raise Exception("Please provide a classifier to check")
                    else:
                        self.model = load(model_path)
                        self.paramDict['model_path'] = model_path
                        self.paramDict['model_type'] = 'sklearn'
                    
                else:
                    self.paramDict['model_type'] = 'sklearn'
                    self.model = model
                    dump(self.model, 'Model/MUT.joblib')
                        
            elif model_type == 'Pytorch':
                if model is None:
                    if model_path == '':
                        raise Exception("Please provide a classifier to check")
                    else:
                        self.paramDict['model_type'] = 'Pytorch'
                        self.paramDict['model_path'] = model_path
                        self.model = Net()
                        self.model = torch.load(model_path)
                        self.model.eval()
                else:
                    self.paramDict['model_type'] = 'Pytorch'
                    self.model = model
                    self.model.eval()
            elif model_type == 'others':
                self.paramDict['model_type'] = 'others'
                self.paramDict['model_path'] = model_path
                self.model = model
                
            else:
                raise Exception("Please provide the type of the model (Pytorch/sklearn)")

            if no_of_train is None:
                self.no_of_train = 1000
            else:
                self.no_of_train = no_of_train
            if train_data_available:
                if train_data_loc == '':
                    raise Exception('Please provide the training data location')
                    sys.exit(1)
                else:
                    if train_ratio is None:
                        self.paramDict['train_ratio'] = 50
                    else:
                        self.paramDict['train_ratio'] = train_ratio
            self.paramDict['no_of_train'] = self.no_of_train
            self.paramDict['train_data_available'] = train_data_available
            self.paramDict['train_data_loc'] = train_data_loc

            try:
                with open('param_dict.csv', 'w', newline='') as csv_file:
                    writer = cv.writer(csv_file)
                    for key, value in self.paramDict.items():
                        writer.writerow([key, value])
            except IOError:
                print("I/O error")

            genData = readXmlFile(self.xml_file)
            genData.funcReadXml()
            gen_oracle = makeOracleData(self.model)
            gen_oracle.funcGenOracle()


class runChecker:
    def __init__(self):
        self.df = pd.read_csv('OracleData.csv')
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

        self.model_type = self.paramDict['model_type']
        if 'model_path' in self.paramDict:
            model_path = self.paramDict['model_path']
            if self.model_type == 'Pytorch':
                self.model = Net()
                self.model = torch.load(model_path)
                self.model.eval()
            elif self.model_type == 'others':
                self.model = get_deepset_model(10)
                self.model = load_model(model_path)    
                deep_we = []
                for i in [1,2,4]:
                    w = self.model.get_layer(index=i).get_weights()
                    deep_we.append(w)
                # load weights
                for i, idx in enumerate([1,2,4]):
                    self.model.get_layer(index=idx).set_weights(deep_we[i])
            else:
                self.model = load(model_path)
        else:
            self.model = load('Model/MUT.joblib')


    def funcCreateOracle(self):
        dfTest = pd.read_csv('TestingData.csv')
        data = dfTest.values
        X = data[:, :-1]
        if self.paramDict['model_type'] == 'Pytorch':
            X = torch.tensor(X, dtype=torch.float32)
            predict_class = []
            for i in range(0, X.shape[0]):
                predict_prob = self.model(X[i].view(-1, X.shape[1]))
                predict_class.append(int(torch.argmax(predict_prob)))
            for i in range(0, X.shape[0]):
                dfTest.loc[i, 'Class'] = predict_class[i]
        else:
            predict_class = self.model.predict(X)
            for i in range(0, X.shape[0]):
                dfTest.loc[i, 'Class'] = predict_class[i]
        dfTest.to_csv('OracleData.csv', index = False, header = True)

    def funcPrediction(self, X, dfCand, testIndx):
        if self.model_type == 'Pytorch':
            X_pred = torch.tensor(X[testIndx], dtype=torch.float32)
            predict_prob = self.model(X_pred.view(-1, X.shape[1]))
            return int(torch.argmax(predict_prob))
        else:
            return self.model.predict(util.convDataInst(X, dfCand, testIndx, 1))

    def func_match_mut_pred(self, X, model, arr_length):
        with open('logicAssert.txt') as f1:
            file_content = f1.readlines()
        file_content = [x.strip() for x in file_content]
        logic2assert.assert_rev(file_content[0])
        import utils.match_mutprediction
        return utils.match_mutprediction.func_match_mut_pred(X, model, arr_length)


    def addModelPred(self):
        dfCexSet = pd.read_csv('CexSet.csv')
        dataCex = dfCexSet.values
        if self.model_type == 'Pytorch':
            X = dataCex[:, :-1]
            X = torch.tensor(X, dtype=torch.float32)
            predict_class=[]
            for i in range(0, X.shape[0]):
                predict_prob = self.model(X[i].view(-1, X.shape[1]))
                predict_class.append(int(torch.argmax(predict_prob)))
        else:
            predict_class = self.model.predict(dataCex[:,:-1])
        for i in range(0, dfCexSet.shape[0]):
            dfCexSet.loc[i, 'Class'] = predict_class[i]
        dfCexSet.to_csv('CexSet.csv', index = False, header = True)

    
    def runWithDNN(self):
        self.no_of_params = int(self.paramDict['no_of_params'])
        retrain_flag = False
        retrain_count = 0
        MAX_CAND_ZERO = 1
        count_cand_zero = 0
        count = 0
        satFlag = False
        start_time = time.time()
        
        while count < self.max_samples:
            print('Retrain count for DNN is:', retrain_count)
            if self.paramDict['nn-library'] == 'Pytorch':
                trainDNN.functrainDNN()
                #print('DNN count is:', count)
                obj_dnl = DNN2logic.ConvertDNN2logic()
                obj_dnl.funcDNN2logic()
            else:
                trainDNN.functrainDNNSklearn()
                # print('DNN count is:', count)
                obj_dnl = DNN2logic.ConvertDNNSklearn2logic()
                obj_dnl.funcDNN2logic()
            util.storeAssumeAssert('DNNSmt.smt2')
            util.addSatOpt('DNNSmt.smt2')
            os.system(r"z3 DNNSmt.smt2 > FinalOutput.txt")
            satFlag = ReadZ3Output.funcConvZ3OutToData(self.df)           
            if not satFlag:
                if count == 0:
                    print('No CEX is found by the checker in the first trial')
                    return 0
                elif (count != 0) and (self.mul_cex == 'True'):
                    dfCexSet = pd.read_csv('CexSet.csv')
                    if round(dfCexSet.shape[0]/self.no_of_params) == 0:
                        print('No CEX is found')
                        return 0
                    print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                    fileResult = open('results.txt', 'a')
                    fileResult.write('\nTotal number of cex found is:'+str(round(dfCexSet.shape[0]/self.no_of_params)))
                    fileResult.close()
                    self.addModelPred()
                    return round(dfCexSet.shape[0]/self.no_of_params) 
                elif (count != 0) and (self.mul_cex == 'False'):
                    print('No Cex is found after '+str(count)+' no. of trials')
                    return 0 
            else:
                df_smt = pd.read_csv('TestDataSMT.csv')
                data_smt = df_smt.values
                X_smt = data_smt[:, :-1]
                if self.func_match_mut_pred(X_smt, self.model, self.no_of_params) == 1 and self.mul_cex == 'False':
                    return
                processCandCex.funcAddCex2CandidateSet()
                processCandCex.funcAddCexPruneCandidateSet4DNN()
                processCandCex.funcCheckCex()

                # Increase the count if no further candidate cex has been found
                dfCand = pd.read_csv('Cand-set.csv')
                if round(dfCand.shape[0] / self.no_of_params) == 0:
                    count_cand_zero += 1
                    if count_cand_zero == MAX_CAND_ZERO:
                        if self.mul_cex == 'True':
                            dfCexSet = pd.read_csv('CexSet.csv')
                            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
                            fileResult = open('results_aware_dnn.txt', 'a')
                            fileResult.write('\nTotal number of cex found is:' + str(round(dfCexSet.shape[0] / self.no_of_params)))
                            fileResult.close()
                            if round(dfCexSet.shape[0] / self.no_of_params) > 0:
                                self.addModelPred()
                            return round(dfCexSet.shape[0] / self.no_of_params) + 1
                        else:
                            print('No CEX is found by the checker')
                            return 0
                else:
                    count = count + round(dfCand.shape[0] / self.no_of_params)

                data = dfCand.values
                X = data[:, :-1]
                y = data[:, -1]
                if dfCand.shape[0] % self.no_of_params == 0:
                    arr_length = dfCand.shape[0]
                else:
                    arr_length = dfCand.shape[0] - 1
                if self.func_match_mut_pred(X, self.model, arr_length) == 1:
                    return
                if (time.time() - start_time) > self.deadline:
                    print("Time out")
                    break
            retrain_count = retrain_count+1

        dfCexSet = pd.read_csv('CexSet.csv')
        if (round(dfCexSet.shape[0] / self.no_of_params) > 0) and (count >= self.max_samples):
            self.addModelPred()
            print('Total number of cex found is:', round(dfCexSet.shape[0] / self.no_of_params))
            print('No. of Samples looked for counter example has exceeded the max_samples limit')
            fileResult = open('results_aware_dnn.txt', 'a')
            fileResult.write('\nTotal number of cex found is:' + str(round(dfCexSet.shape[0] / self.no_of_params)))
            fileResult.close()
        else:
            print('No counter example has been found')

    def runPropCheck(self):
        retrain_flag = False
        retrain_count = 0
        MAX_CAND_ZERO = 1
        count_cand_zero = 0
        count = 0
        satFlag = False
        self.max_samples = int(self.paramDict['max_samples'])
        self.no_of_params = int(self.paramDict['no_of_params'])
        self.mul_cex = self.paramDict['mul_cex_opt']
        self.deadline = int(self.paramDict['deadlines'])
        white_box = self.paramDict['white_box_model']
        start_time = time.time()
        
        if white_box == 'DNN':
            self.runWithDNN()
        else:    
            while count < self.max_samples:
                #print('count is:', count)
                print('Retrain count for Decision tree is:', retrain_count)
                train_obj = trainDecTree()
                tree = train_obj.functrain(self.model, count)
                tree2Logic.functree2LogicMain(tree, self.no_of_params)
                util.storeAssumeAssert('DecSmt.smt2')
                util.addSatOpt('DecSmt.smt2')
                if self.paramDict['solver'] == 'z3':
                    #print('running z3')
                    os.system(r"z3 DecSmt.smt2 > FinalOutput.txt")
                elif self.paramDict['solver'] == 'yices':
                    #print('running yieces')
                    os.system(r"yices-smt2 DecSmt.smt2 > FinalOutput.txt")
                elif self.paramDict['solver'] == 'cvc':

                    os.system(r"cvc4 DecSmt.smt2 > FinalOutput.txt")
                #open("sat_out.txt", "a").writelines([l for l in open("FinalOutput.txt").readlines()])
                satFlag = ReadZ3Output.funcConvZ3OutToData(self.df)
                if not satFlag:
                    if count == 0:
                        print('No CEX is found by the checker at the first trial')
                        time.time() - start_time
                        return 0
                    elif (count != 0) and (self.mul_cex == 'True'):
                        dfCexSet = pd.read_csv('CexSet.csv')
                        if round(dfCexSet.shape[0]/self.no_of_params) == 0:
                            print('No CEX is found')
                            return 0
                        print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                        fileResult = open('results_aware_decision-tree.txt', 'a')
                        fileResult.write('\nTotal Time required is:' + str(time.time() - start_time))
                        fileResult.write('\nTotal number of cex found is:' + str(
                            round(dfCexSet.shape[0] / self.no_of_params)))
                        fileResult.close()
                        self.addModelPred()
                        return round(dfCexSet.shape[0]/self.no_of_params) 
                    elif (count != 0) and (self.mul_cex == 'False'):
                        print('No Cex is found after '+str(count)+' no. of trials')
                        return 0
                else:
                    df_smt = pd.read_csv('TestDataSMT.csv')
                    data_smt = df_smt.values
                    X_smt = data_smt[:, :-1]
                    if self.func_match_mut_pred(X_smt, self.model, self.no_of_params) == 1 and self.mul_cex == 'False':
                        return 1
                    processCandCex.funcAddCex2CandidateSet()
                    processCandCex.funcAddCexPruneCandidateSet(tree)
                    processCandCex.funcCheckCex()

                    #Increase the count if no further candidate cex has been found
                    dfCand = pd.read_csv('Cand-set.csv')
                    if round(dfCand.shape[0]/self.no_of_params) == 0:
                        count_cand_zero += 1
                        if count_cand_zero == MAX_CAND_ZERO:
                            if self.mul_cex == 'True':
                                dfCexSet = pd.read_csv('CexSet.csv')
                                print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                                fileResult = open('results_aware_decision-tree.txt', 'a')
                                fileResult.write('\nTotal Time required is:' + str(time.time() - start_time))
                                fileResult.write('\nTotal number of cex found is:' + str(
                                    round(dfCexSet.shape[0] / self.no_of_params)))
                                fileResult.close()
                                if round(dfCexSet.shape[0]/self.no_of_params) > 0:
                                    self.addModelPred()
                                return round(dfCexSet.shape[0]/self.no_of_params) + 1
                            else:
                                print('No CEX is found by the checker')
                                return 0
                    else:
                        count = count + round(dfCand.shape[0]/self.no_of_params)

                    data = dfCand.values
                    X = data[:, :-1]
                    y = data[:, -1]    
                    if dfCand.shape[0] % self.no_of_params == 0:
                        arr_length = dfCand.shape[0]
                    else:
                        arr_length = dfCand.shape[0]-1
                    if self.func_match_mut_pred(X, self.model, arr_length) == 1:
                        return 1

                    if (time.time() - start_time) > self.deadline:
                        print("Time out")
                        break
                retrain_count = retrain_count+1
            dfCexSet = pd.read_csv('CexSet.csv')
            if (round(dfCexSet.shape[0]/self.no_of_params) > 0) and (count >= self.max_samples):
                self.addModelPred()
                print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
                fileResult = open('results_aware_decision-tree.txt', 'a')
                fileResult.write('\nTotal Time required is:' + str(time.time() - start_time))
                fileResult.write('\nTotal number of cex found is:'+str(round(dfCexSet.shape[0] / self.no_of_params)))
                fileResult.close()
                print('No. of Samples looked for counter example has exceeded the max_samples limit')
            else:
                print('No counter example has been found')


def Assume(*args):
    grammar = Grammar(
        r"""

    expr        = expr1 / expr2 / expr3 /expr4 /expr5 / expr6 /expr7
    expr1       = expr_dist1 logic_op num_log
    expr2       = expr_dist2 logic_op num_log
    expr3       = classVar ws logic_op ws value
    expr4       = classVarArr ws logic_op ws value
    expr5       = classVar ws logic_op ws classVar
    expr6       = classVarArr ws logic_op ws classVarArr
    expr7       = "True"
    expr_dist1  = op_beg?abs?para_open classVar ws arith_op ws classVar para_close op_end?
    expr_dist2  = op_beg?abs?para_open classVarArr ws arith_op ws classVarArr para_close op_end?
    classVar    = variable brack_open number brack_close
    classVarArr = variable brack_open variable brack_close
    para_open   = "("
    para_close  = ")"
    brack_open  = "["
    brack_close = "]"
    variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    logic_op    = ws (geq / leq / eq / neq / and / lt / gt) ws
    op_beg      = number arith_op
    op_end      = arith_op number
    arith_op    = (add/sub/div/mul)
    abs         = "abs"
    add         = "+"
    sub         = "-"
    div         = "/"
    mul         = "*"
    lt          = "<"
    gt          = ">"
    geq         = ">="
    leq         = "<="
    eq          = "="
    neq         = "!="
    and         = "&"
    ws          = ~"\s*"
    value       = ~"\d+"
    num_log     = ~"[+-]?([0-9]*[.])?[0-9]+"
    number      = ~"[+-]?([0-9]*[.])?[0-9]+"
    """
    )

    tree = grammar.parse(args[0])
    # print(tree)

    assumeVisitObj = assume2logic.AssumptionVisitor()
    if len(args) == 3:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.storeArr(args[2])
        assumeVisitObj.visit(tree)
    elif len(args) == 2:
        assumeVisitObj.storeInd(args[1])
        assumeVisitObj.visit(tree)
    elif len(args) == 1:
        assumeVisitObj.visit(tree)


            

def Assert(*args):    
    grammar = Grammar(
    r"""
    expr        = expr13/ expr1 / expr2/ expr3/ expr4/ expr5 / expr6/ expr7/ expr8 /expr9/ expr10/ expr11/ expr12
    expr1       = classVar ws operator ws number
    expr2       = classVar ws operator ws classVar
    expr3       = classVar mul_cl_var ws operator ws neg? classVar mul_cl_var
    expr4       = classVar ws? operator ws? min_symbol brack_open variable brack_close
    expr5       = classVar ws? operator ws? max_symbol brack_open variable brack_close
    expr6       = abs? brack_open classVar ws? arith_op1 ws? classVar brack_close ws? operator ws? (number arith_op2)?("const" arith_op2)?
     "manhattan_distance" brack_open variable "," variable brack_close
    expr7       =  classVar ws? operator ws? "const"
    expr8       = "symmetric1" ws? brack_open classVar brack_close
    expr9       = "symmetric2" ws? brack_open classVar brack_close
    expr10      = min_symbol brack_open variable brack_close ws? operator ws? classVar ws? operator ws? max_symbol brack_open variable brack_close
    expr11      = classVar ws? operator ws? "annihilator"
    expr12      = "model.predict(x+y) == model.predict(x)+model.predict(y)"
    expr13      = classVar ws? operator ws? number ws? arith_op1 ws? classVar
    classVar    = class_pred brack_open variable brack_close
    model_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    class_pred  = model_name classSymbol
    classSymbol = ~".predict"
    const       = "const"
    min_symbol  = "min"
    max_symbol  = "max"
    abs         = "abs"
    brack_open  = "("
    brack_close = ")"
    variable    = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    brack3open  = "["
    brack3close = "]"
    class_name  = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
    mul_cl_var  = brack3open class_name brack3close
    operator    = ws (geq / leq / eq / gt/ lt/ neq / and/ implies) ws
    arith_op1    = (add/sub/div/mul)
    arith_op2    = (add/sub/div/mul)
    add         = "+"
    sub         = "-"
    div         = "/"
    mul         = "*"
    lt          = "<"
    gt          = ">"
    geq         = ~">="
    implies     = "=>"
    neg         = "~"
    leq         = "=<"
    eq          = "=="
    neq         = "!="
    and         = "&"
    ws          = ~"\s*"
    number      = ~"[+-]?([0-9]*[.])?[0-9]+"
    """
    )
      
    tree = grammar.parse(args[0])
    assertVisitObj = assert2logic.AssertionVisitor()
    assertVisitObj.visit(tree)
    
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    if paramDict['multi_label'] == 'True':
        start_time = time.time()
        obj_multi = multiLabelMain.runChecker()
        obj_multi.runPropCheck()
        print('time required is', time.time()-start_time)
    else:
        obj_faircheck = runChecker()
        start_time = time.time()
        if obj_faircheck.runPropCheck() == 1:
            print('time required is', time.time()-start_time)
            if os.path.exists('assumeStmnt.txt'):
                os.remove('assumeStmnt.txt')
            os.remove('assertStmnt.txt')
            return True

    if os.path.exists('assumeStmnt.txt'):
        os.remove('assumeStmnt.txt')
    #if os.path.exists('TestSet.csv'):
    #    os.remove('TestSet.csv')
    os.remove('assertStmnt.txt')
    return False


class trainDecTree:
    def __init__(self):
        pass

    def functrain(self, model_mut, count):
        score = 0
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            paramDict = dict(reader)
        xml_file = paramDict['xml_file']
        depth_list = [i for i in range(2, 500001)]
        random_state_list = [i for i in range(2, 900001)]
        samples_leaf_list = [i for i in range(2, 70)]
        samples_split_list = [i for i in range(2, 70)]
        if paramDict['regression'] == 'yes':
            itertaion_count = 0
            score_list = []
            classifier_list = []
            tree_model = DecisionTreeRegressor(max_depth= 2)

            df = pd.read_csv('OracleData.csv')
            data = df.values
            X = data[:, :-1]
            y = data[:, -1]
            model = tree_model.fit(X, y)
            '''
            while score < 0.80:
                df = pd.read_csv('OracleData.csv')
                data = df.values
                X = data[:, :-1]
                y = data[:, -1]
                param_space = {"max_depth": [i for i in depth_list],
                               "criterion": ['mse', 'friedman_mse', 'mae'],
                               #"splitter": ['best', 'random'],
                               #"max_features": ['auto', 'sqrt', 'log2', None],
                               #"min_samples_leaf": [i for i in samples_leaf_list],
                               #"min_samples_split": [i for i in samples_split_list],
                               #"max_leaf_nodes": [i for i in random_state_list],
                               "ccp_alpha": loguniform(1e-7, 1000),
                               "random_state": [i for i in random_state_list]
                               }

                tree_rand_search = RandomizedSearchCV(tree_model, param_space, n_iter=50,
                                            scoring="r2", verbose=True, cv=5,
                                            n_jobs=-1, random_state=50)
                tree_rand_search.fit(X, y)
                score = tree_rand_search.best_score_
                score_list.append(score)
                classifier_list.append(tree_rand_search.best_estimator_)
                itertaion_count += 1
                model = tree_rand_search.best_estimator_
                if count > 0:
                    break
                if itertaion_count == 3:
                    element = max([i for i in score_list])
                    index = score_list.index(element)
                    model = classifier_list[index]
                    break

                if score < 0.70:
                    genData = readXmlFile(xml_file)
                    genData.funcReadXml()
                    gen_oracle = makeOracleData(model_mut)
                    gen_oracle.funcGenOracle()
                #print(tree_rand_search.best_score_)
            '''
        else:
            tree_model = DecisionTreeClassifier()
            df = pd.read_csv('OracleData.csv')
            data = df.values
            X = data[:, :-1]
            y = data[:, -1]
            model = tree_model.fit(X, y)
            '''
            while score < 0.80:
                df = pd.read_csv('OracleData.csv')
                data = df.values
                X = data[:, :-1]
                y = data[:, -1]

                param_space = {"max_depth": [i for i in depth_list],
                               "criterion": ['gini', 'entropy'],
                               "splitter": ['best', 'random'],
                               "max_features": ['auto', 'sqrt', 'log2', None],
                               "max_leaf_nodes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                               "min_samples_leaf": [1, 2, 3, 4, 5],
                               "min_samples_split": [2, 3, 4, 5],
                               "random_state": [i for i in random_state_list]
                               }

                tree_rand_search = RandomizedSearchCV(tree_model, param_space, n_iter=50,
                                                      scoring="accuracy", verbose=True, cv=5,
                                                      n_jobs=-1, random_state=42)
                tree_rand_search.fit(X, y)
                score = tree_rand_search.best_score_
                model = tree_rand_search.best_estimator_
                if count > 0:
                    break
                if score < 0.80:
                    genData = readXmlFile(xml_file)
                    genData.funcReadXml()
                    gen_oracle = makeOracleData(model_mut)
                    gen_oracle.funcGenOracle()
                #print(tree_rand_search.best_score_)
            '''
        dump(model, 'Model/dectree_approx.joblib')
        return model
