import pandas as pd
import csv as cv
import numpy as np
from ast import literal_eval
import re
from sklearn.tree import _tree


def tree_to_code(tree, feature_names, paramDict):
    f = open('TreeOutput.txt', 'w')
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    f.write("def tree({}):".format(", ".join(feature_names)))
    f.write("\n")

    def recurse(node, depth):
        indent = "  " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            f.write("{}if {} <= {}:".format(indent, name, threshold))
            f.write("\n")
            f.write("{}".format(indent)+"{")
            f.write("\n")
            
            recurse(tree_.children_left[node], depth + 1)
            f.write("{}".format(indent)+"}")
            f.write("\n")

            f.write("{}else:  # if {} > {}".format(indent, name, threshold))
            f.write("\n")

            f.write("{}".format(indent)+"{")
            f.write("\n")
            
            recurse(tree_.children_right[node], depth + 1)

            f.write("{}".format(indent)+"}")
            f.write("\n")
            
        else:
            if paramDict['regression'] == 'yes':
                f.write("{}return {}".format(indent, tree_.value[node][0]))
            else:
                f.write("{}return {}".format(indent, np.argmax(tree_.value[node][0])))
            f.write("\n")

    recurse(0, 1)
    f.close() 


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def funcConvBranch(single_branch, dfT, rep):
    with open('feNameType.csv') as csv_file:
        reader = cv.reader(csv_file)
        feName_type = dict(reader)
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)

    f = open('DecSmt.smt2', 'a') 
    f.write("(assert (=> (and ")
    for i in range(0, len(single_branch)):
        temp_Str = single_branch[i]
        if 'if' in temp_Str:
            for j in range (0, dfT.columns.values.shape[0]):
                if dfT.columns.values[j] in temp_Str:
                    fe_name = str(dfT.columns.values[j])
                    fe_index = j
            data_type = feName_type[fe_name]

            if '<=' in temp_Str:
                sign = '<='
            elif '<=' in temp_Str:
                sign = '>'    
            elif '>' in temp_Str:
                sign = '>'
            elif '>=' in temp_Str:
                sign = '>='
            digit = float(re.search(r'[sign ][+-]?([0-9]*[.])?[0-9]+', temp_Str).group(0))
            #digit = format(digit, '.10f')
            if 'int' in data_type:
                digit = int(round(digit))
            digit = str(digit)
            f.write("(" + sign + " "+ fe_name +str(rep)+" " + digit +") ") 

        elif 'return' in temp_Str:
            digit_class = float(re.search(r'[+-]?([0-9]*[.])?[0-9]+', temp_Str).group(0))
            #digit_class = format(digit_class, '.7f')
            if paramDict['regression'] == 'no':
                digit_class = int(round(digit_class))
            digit_class = str(digit_class)
            f.write(") (= Class"+str(rep)+" "+digit_class +")))")
            f.write('\n')
    f.close()


def funcGetBranch(sinBranch, dfT, rep):
    for i in range (0, len(sinBranch)):
        tempSt = sinBranch[i]
        if 'return' in tempSt:
            funcConvBranch(sinBranch, dfT, rep)


def funcGenBranch(dfT, rep, paramDict):
    with open('TreeOutput.txt') as f1:
        file_content = f1.readlines()
    file_content = [x.strip() for x in file_content]
    f1.close()
    noOfLines = file_len('TreeOutput.txt')
    temp_file_cont = ["" for x in range(noOfLines)]
    
    i = 1
    k = 0
    while i < noOfLines:
        j = k-1
        if temp_file_cont[j] == '}':
            funcGetBranch(temp_file_cont, dfT, rep)
            while True:
                if temp_file_cont[j] == '{':
                    temp_file_cont[j] = ''
                    temp_file_cont[j-1] = ''
                    j = j-1
                    break  
                elif j>=0:
                    temp_file_cont[j] = ''
                    j = j-1
            k = j    
            
        else:    
            temp_file_cont[k] = file_content[i]
            k = k+1
            i = i+1
  
    if 'return' in file_content[1]:
        digit = float(re.search(r'[+-]?([0-9]*[.])?[0-9]+', file_content[1]).group(0))
        if paramDict['regression'] == 'no':
            digit = int(round(digit))
        f = open('DecSmt.smt2', 'a')
        f.write("(assert (= Class"+str(rep)+" "+str(digit)+"))")
        f.write("\n")
        f.close()
    else:    
        funcGetBranch(temp_file_cont, dfT, rep)


def funcConv(dfT, no_of_instances, paramDict):
    try:
        # Read feature types
        with open('feNameType.csv') as csv_file:
            reader = cv.reader(csv_file)
            feName_type = {}
            for row in reader:
                if len(row) == 2:  # Only process valid key-value pairs
                    feName_type[row[0]] = row[1]
        
        # Read min values
        with open('feMinValue.csv') as csv_file:
            reader = cv.reader(csv_file)
            feMinVal = {}
            for row in reader:
                if len(row) == 2:  # Only process valid key-value pairs
                    feMinVal[row[0]] = float(row[1])
        
        # Read max values
        with open('feMaxValue.csv') as csv_file:
            reader = cv.reader(csv_file)
            feMaxVal = {}
            for row in reader:
                if len(row) == 2:  # Only process valid key-value pairs
                    feMaxVal[row[0]] = float(row[1])
    except (FileNotFoundError, IOError) as e:
        raise Exception(f"Error reading feature specification files: {str(e)}")
    except ValueError as e:
        raise Exception(f"Error parsing feature values: {str(e)}")

    bound_cex = literal_eval(paramDict['bound_cex'])
    bound_list = eval(paramDict['bound_list'])
    bound_all = eval(paramDict['bound_all_features'])

    f = open('DecSmt.smt2', 'w')
    if paramDict['solver'] != 'z3':
        f.write('(set-logic QF_LIRA)\n')
    if paramDict['solver'] == 'cvc':
        f.write('(set-option :produce-models true)\n')
    #f.write('(set-option :pp.decimal true) \n (set-option :pp.decimal_precision 8) \n')
    for j in range(0, no_of_instances):
        for i in range (0, dfT.columns.values.shape[0]):
            tempStr = dfT.columns.values[i]
            fe_type = feName_type[tempStr]
            min_val = feMinVal[tempStr]
            max_val = feMaxVal[tempStr]
        
            if 'int' in fe_type:
                f.write("(declare-fun " + tempStr+str(j)+ " () Int)")
                f.write('\n')
                #adding range
                if bound_cex and tempStr in bound_list:
                    f.write("(assert (and (>= "+tempStr+str(j)+" "+str(int(min_val))+")"+" "+"(<= "+tempStr+str(j)+" "+str(int(max_val))+")))")
                elif bound_cex and bound_all:
                    f.write("(assert (and (>= " + tempStr + str(j) + " " + str(
                        int(min_val)) + ")" + " " + "(<= " + tempStr + str(j) + " " + str(int(max_val)) + ")))")
                f.write('\n')
            elif('float' in fe_type):
                f.write("(declare-fun " + tempStr+str(j)+ " () Real)")
                f.write('\n')
                #Adding range
                if bound_cex and tempStr in bound_list:
                    #f.write("(assert (and (>= "+tempStr+str(j)+" "+str(format(min_val, '.3f'))+")"+" "+"(<= "+tempStr+str(j)+" "+str(format(max_val, '.3f'))+")))")
                    f.write("(assert (and (>= " + tempStr + str(j) + " " + str(min_val) + ")" + " " + "(<= " + tempStr +
                            str(j) + " " + str(max_val) + ")))")
                elif bound_cex and bound_all:
                    f.write("(assert (and (>= " + tempStr + str(j) + " " + str(min_val) + ")" + " " + "(<= " + tempStr +
                            str(j) + " " + str(max_val) + ")))")
                f.write('\n') 
        f.write("; "+str(j)+"th element")
        f.write('\n')
    '''
    for i in range(0, no_of_instances):
        if 'int' in feName_type['Class']:
            f.write("(declare-fun Class"+str(i)+ " () Int)")
        else:
            f.write("(declare-fun Class" + str(i) + " () Real)")
        f.write('\n')
    '''
    #Writing the functions for computing absolute integer & real value
    f.write('(define-fun absoluteInt ((x Int)) Int \n')
    f.write('  (ite (>= x 0) x (- x))) \n')

    f.write('(define-fun absoluteReal ((x Real)) Real \n')
    f.write('  (ite (>= x 0) x (- x))) \n')
    f.write('(declare-fun constnt () Real)\n')
    f.write('(declare-fun temp () Real)\n')
    f.close()
    
    #Calling function to get the branch and convert it to z3 form,  creating alias
    for i in range(0, no_of_instances):  
        f = open('DecSmt.smt2', 'a')
        f.write('\n;-----------'+str(i)+'-----------number instance-------------- \n')
        f.close()
        funcGenBranch(dfT, i, paramDict)
    

def functree2LogicMain(tree, no_of_instances):
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    df = pd.read_csv('OracleData.csv')
    tree_to_code(tree, df.columns, paramDict)
    funcConv(df, no_of_instances, paramDict)
