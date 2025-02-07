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

        for k in range(0, len(self.nameArr)):
            fe_type = self.typeArr[k]
            if 'int' in fe_type:
                # Convert min and max values to integers for integer features
                min_val = int(self.minArr[k])
                max_val = int(self.maxArr[k])
                tempData[0][k] = rd.randint(min_val, max_val)
            else:
                # Keep as float for non-integer features
                tempData[0][k] = rd.uniform(self.minArr[k], self.maxArr[k])
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
        import sys
        
        def log(msg):
            print(msg, flush=True)
            sys.stdout.flush()
        
        try:
            log("Starting test data generation...")
            log(f"Number of features: {len(self.nameArr)}")
            log(f"Feature names: {self.nameArr}")
            log(f"Feature types: {self.typeArr}")
            log(f"Min values: {self.minArr}")
            log(f"Max values: {self.maxArr}")
            
            tst_pm = int(self.paramDict['no_of_train'])
            log(f"Generating {tst_pm} test samples...")
            
            # Generate test matrix without Class column initially
            testMatrix = np.zeros((tst_pm + 1, len(self.nameArr)), dtype=object)
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
                    if i % 100 == 0:
                        log(f"Generated {i} samples...")

            log("Writing test data to CSV...")
            with open('TestingData.csv', 'w', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerow(self.nameArr)  # Write feature names without Class
                writer.writerows(testMatrix)
            log("Test data generation complete!")

            if self.paramDict['train_data_available'] == 'True':
                log("Processing training data...")
                dfTrainData = pd.read_csv(self.paramDict['train_data_loc'])
                self.generateTestTrain(dfTrainData, int(self.paramDict['train_ratio']))
            
            # Create empty TestSet.csv and CexSet.csv with proper headers
            with open('TestSet.csv', 'w', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerow(self.nameArr + ['Class'])  # Add Class for output files
            with open('CexSet.csv', 'w', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerow(self.nameArr + ['Class'])  # Add Class for output files
                
        except Exception as e:
            log(f"Error in test data generation: {str(e)}")
            raise

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
        self.feature_names = []
        self.feature_types = []
        self.min_values = []
        self.max_values = []

    def funcReadXml(self):
        import xml.etree.ElementTree as ET
        import sys
        
        def log(msg):
            print(msg, flush=True)
            sys.stdout.flush()

        try:
            log(f"Reading XML file: {self.fileName}")
            
            # Parse the XML file
            tree = ET.parse(self.fileName)
            root = tree.getroot()
            log("Successfully parsed XML file")

            # Process each Input element
            input_count = 0
            for input_elem in root.findall('Input'):
                input_count += 1
                try:
                    # Extract feature name
                    name = input_elem.find('Feature-name').text
                    self.feature_names.append(name)

                    # Extract feature type
                    type_str = input_elem.find('Feature-type').text
                    self.feature_types.append(type_str)

                    # Extract min and max values
                    value_elem = input_elem.find('Value')
                    min_val = float(value_elem.find('minVal').text)
                    max_val = float(value_elem.find('maxVal').text)

                    self.min_values.append(min_val)
                    self.max_values.append(max_val)
                    
                except Exception as e:
                    log(f"Error processing input element {input_count}: {str(e)}")
                    raise

            log(f"Processed {input_count} input elements")
            log(f"Feature names: {self.feature_names}")
            log(f"Feature types: {self.feature_types}")

            # Create the feature specifications DataFrame
            data = {
                'Feature': self.feature_names,
                'Type': self.feature_types,
                'MinValue': self.min_values,
                'MaxValue': self.max_values
            }
            df = pd.DataFrame(data)
            log("Created feature specifications DataFrame")

            # Save to CSV for later use
            df.to_csv('DataFeatureSpec.csv', index=False)
            log("Saved feature specifications to DataFeatureSpec.csv")

            return True

        except Exception as e:
            log(f"Error reading XML file: {str(e)}")
            import traceback
            log(traceback.format_exc())
            return False


class makeOracleData:

    def __init__(self, model):
        self.model = model
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)

    def funcGenOracle(self):
        try:
            # Read test data
            dfTest = pd.read_csv('TestingData.csv')
            dataTest = dfTest.values
            
            # Initialize predictions array
            predict_list = np.zeros((1, dfTest.shape[0]))
            X = dataTest  # Use all columns since there's no Class column

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
                    dfTest['Class'] = predict_class
                else:
                    try:
                        predict_class = self.model.predict(X)
                        if self.paramDict['regression'] == 'yes':
                            dfTest['Class'] = predict_class
                        else:
                            dfTest['Class'] = predict_class.astype(int)
                    except Exception as e:
                        raise Exception(f"Failed to make predictions: {str(e)}")

            # Save the oracle data
            dfTest.to_csv('OracleData.csv', index=False)
            
        except pd.errors.EmptyDataError:
            raise Exception("TestingData.csv is empty")
        except FileNotFoundError:
            raise Exception("TestingData.csv not found")
        except Exception as e:
            raise Exception(f"Error in oracle data generation: {str(e)}")


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
                        try:
                            # Load the model architecture first
                            from main import SimpleNN
                            # Get input size from the data
                            df_specs = pd.read_csv('DataFeatureSpec.csv')
                            input_size = len(df_specs['Feature'])
                            # Initialize model with correct architecture
                            self.model = SimpleNN(input_size)
                            # Load the state dict
                            self.model.load_state_dict(torch.load(model_path))
                            self.model.eval()  # Set model to evaluation mode
                        except Exception as e:
                            raise Exception(f"Failed to load PyTorch model: {str(e)}")
                else:
                    self.paramDict['model_type'] = 'Pytorch'
                    self.model = model
                    self.model.eval()  # Set model to evaluation mode
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

            # Step 1: Read XML and create feature specifications
            xml_reader = readXmlFile(self.xml_file)
            if not xml_reader.funcReadXml():
                raise Exception("Failed to read XML file")
            
            # Step 2: Generate test data using feature specifications
            try:
                df_specs = pd.read_csv('DataFeatureSpec.csv')
                # Create separate CSV files for feature types, min values, and max values
                feature_type_dict = dict(zip(df_specs['Feature'], df_specs['Type']))
                feature_min_dict = dict(zip(df_specs['Feature'], df_specs['MinValue']))
                feature_max_dict = dict(zip(df_specs['Feature'], df_specs['MaxValue']))
                
                # Write feature types
                with open('feNameType.csv', 'w', newline='') as csv_file:
                    writer = cv.writer(csv_file)
                    for key, value in feature_type_dict.items():
                        writer.writerow([key, value])
                
                # Write min values
                with open('feMinValue.csv', 'w', newline='') as csv_file:
                    writer = cv.writer(csv_file)
                    for key, value in feature_min_dict.items():
                        writer.writerow([key, value])
                
                # Write max values
                with open('feMaxValue.csv', 'w', newline='') as csv_file:
                    writer = cv.writer(csv_file)
                    for key, value in feature_max_dict.items():
                        writer.writerow([key, value])
                
                data_generator = generateData(df_specs['Feature'].tolist(), df_specs['Type'].tolist(), 
                                           df_specs['MinValue'].tolist(), df_specs['MaxValue'].tolist())
                data_generator.funcGenerateTestData()
            except Exception as e:
                raise Exception(f"Failed to generate test data: {str(e)}")
            
            # Step 3: Create oracle data using the test data
            try:
                gen_oracle = makeOracleData(self.model)
                gen_oracle.funcGenOracle()
            except Exception as e:
                raise Exception(f"Failed to create oracle data: {str(e)}")


class runChecker:
    def __init__(self):
        try:
            # Read OracleData.csv and ensure Class column exists
            self.df = pd.read_csv('OracleData.csv')
            if 'Class' not in self.df.columns:
                raise Exception("OracleData.csv is missing the 'Class' column")
            
            # Load parameters from param_dict.csv
            try:
                with open('param_dict.csv') as csv_file:
                    reader = cv.reader(csv_file)
                    self.paramDict = {}
                    for row in reader:
                        if len(row) == 2:  # Only process valid key-value pairs
                            self.paramDict[row[0]] = row[1]
            except (FileNotFoundError, IOError) as e:
                raise Exception(f"Error reading param_dict.csv: {str(e)}")

            self.model_type = self.paramDict.get('model_type')
            if not self.model_type:
                raise Exception("model_type not found in param_dict.csv")

            if 'model_path' in self.paramDict:
                model_path = self.paramDict['model_path']
                if self.model_type == 'Pytorch':
                    try:
                        # Load the model architecture first
                        from main import SimpleNN
                        # Get input size from the data
                        df_specs = pd.read_csv('DataFeatureSpec.csv')
                        input_size = len(df_specs['Feature'])
                        # Initialize model with correct architecture
                        self.model = SimpleNN(input_size)
                        # Load the state dict
                        self.model.load_state_dict(torch.load(model_path))
                        self.model.eval()  # Set model to evaluation mode
                    except FileNotFoundError:
                        raise Exception(f"Model file not found at {model_path}")
                    except Exception as e:
                        raise Exception(f"Failed to load PyTorch model: {str(e)}")
                elif self.model_type == 'others':
                    try:
                        self.model = get_deepset_model(10)
                        self.model = load_model(model_path)    
                        deep_we = []
                        for i in [1,2,4]:
                            w = self.model.get_layer(index=i).get_weights()
                            deep_we.append(w)
                        # load weights
                        for i, idx in enumerate([1,2,4]):
                            self.model.get_layer(index=idx).set_weights(deep_we[i])
                    except Exception as e:
                        raise Exception(f"Failed to load other model type: {str(e)}")
                else:
                    try:
                        self.model = load(model_path)
                    except Exception as e:
                        raise Exception(f"Failed to load model from {model_path}: {str(e)}")
            else:
                try:
                    self.model = load('Model/MUT.joblib')
                except Exception as e:
                    raise Exception(f"Failed to load default model from Model/MUT.joblib: {str(e)}")
                    
        except pd.errors.EmptyDataError:
            raise Exception("OracleData.csv is empty")
        except FileNotFoundError:
            raise Exception("OracleData.csv not found")
        except Exception as e:
            raise Exception(f"Error initializing runChecker: {str(e)}")

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
                    arr_length = dfCand.shape[0] - 1
                if self.func_match_mut_pred(X, self.model, arr_length) == 1:
                    return
                if (time.time() - start_time) > self.deadline:
                    print("Time out")
                    break
            retrain_count = retrain_count+1

        dfCexSet = pd.read_csv('CexSet.csv')
        if (round(dfCexSet.shape[0]/self.no_of_params) > 0) and (count >= self.max_samples):
            self.addModelPred()
            print('Total number of cex found is:', round(dfCexSet.shape[0]/self.no_of_params))
            print('No. of Samples looked for counter example has exceeded the max_samples limit')
            fileResult = open('results_aware_dnn.txt', 'a')
            fileResult.write('\nTotal number of cex found is:' + str(round(dfCexSet.shape[0]/self.no_of_params)))
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
    expr        = expr1 / expr2 / expr3 / expr4 / expr5 / expr6 / expr7 / expr8
    expr1       = expr_dist1 logic_op num_log
    expr2       = expr_dist2 logic_op num_log
    expr3       = classVar ws logic_op ws value
    expr4       = classVarArr ws logic_op ws value
    expr5       = classVar ws logic_op ws classVar
    expr6       = classVarArr ws logic_op ws classVarArr
    expr7       = "True"
    expr8       = method_call ws logic_op ws method_call
    expr_dist1  = op_beg?abs?para_open classVar ws arith_op ws classVar para_close op_end?
    expr_dist2  = op_beg?abs?para_open classVarArr ws arith_op ws classVarArr para_close op_end?
    classVar    = variable brack_open number brack_close
    classVarArr = variable brack_open variable brack_close
    method_call = object_name dot method_name para_open variable para_close
    object_name = ~"[a-zA-Z_][a-zA-Z0-9_]*"
    method_name = ~"[a-zA-Z_][a-zA-Z0-9_]*"
    dot         = "."
    para_open   = "("
    para_close  = ")"
    brack_open  = "["
    brack_close = "]"
    variable    = ~"[a-zA-Z_][a-zA-Z0-9_]*"
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
    ws          = ~"[ \t\n\r]*"
    value       = ~"[0-9]+"
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
    expr        = expr1 / expr2 / expr3 / expr4 / expr5 / expr6 / expr7 / expr8
    expr1       = expr_dist1 logic_op num_log
    expr2       = expr_dist2 logic_op num_log
    expr3       = classVar ws logic_op ws value
    expr4       = classVarArr ws logic_op ws value
    expr5       = classVar ws logic_op ws classVar
    expr6       = classVarArr ws logic_op ws classVarArr
    expr7       = "True"
    expr8       = method_call ws logic_op ws method_call
    expr_dist1  = op_beg?abs?para_open classVar ws arith_op ws classVar para_close op_end?
    expr_dist2  = op_beg?abs?para_open classVarArr ws arith_op ws classVarArr para_close op_end?
    classVar    = variable brack_open number brack_close
    classVarArr = variable brack_open variable brack_close
    method_call = object_name dot method_name para_open variable para_close
    object_name = ~"[a-zA-Z_][a-zA-Z0-9_]*"
    method_name = ~"[a-zA-Z_][a-zA-Z0-9_]*"
    dot         = "."
    para_open   = "("
    para_close  = ")"
    brack_open  = "["
    brack_close = "]"
    variable    = ~"[a-zA-Z_][a-zA-Z0-9_]*"
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
    eq          = "=="
    neq         = "!="
    and         = "&"
    ws          = ~"[ \t\n\r]*"
    value       = ~"[0-9]+"
    num_log     = ~"[+-]?([0-9]*[.])?[0-9]+"
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
