

from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
from itertools import groupby
import csv as cv
import re, sys
import pandas as pd



class AssertionVisitor(NodeVisitor):
    
    def __init__(self):
        self.currentClass = []
        self.modelVarList = []
        self.classNameList = []
        self.currentOperator = ""
        self.current_arith_operator2 = ""
        self.current_arith_operator1 = ""
        self.negOp = ""
        self.varList = []
        self.mydict = {}
        self.varMap = {}
        self.feVal = 0
        self.count = 0
        self.const = False
        self.dfOracle = pd.read_csv('OracleData.csv')
        with open('dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.mydict = dict(reader)  
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramDict = dict(reader)
        with open('feNameType.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.fename_type = dict(reader)
        with open('param_list.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.instance_dict = dict(reader)
    
    def generic_visit(self, node, children):
        pass
    
    def visit_classVar(self, node, children):
        if(self.mydict['no_mapping'] == 'True'):
            pass
        else:
            for el in self.varList:
                if(el in node.text):
                    if(self.mydict['no_assumption'] == 'False'):
                        className = 'Class'+str(self.mydict[el])
                    else:
                        className = 'Class'+str(self.count)
            self.currentClass.append(className)
        
    def visit_neg(self, node, children):
        self.negOp = node.text
        
    def visit_model_name(self, node, children):
        self.modelVarList.append(node.text)
        
    def visit_class_name(self, node, children):
        if(node.text in self.dfOracle.columns.values):
            self.classNameList.append(node.text)
        else:
            raise Exception('Class name '+str(node.text)+' do not exist')
        
    def visit_variable(self, node, children):
        if(self.mydict['no_mapping'] == 'True'):
            pass
        else:
            self.varList.append(node.text)
            if(self.mydict['no_assumption'] == 'False'):

                num = str(int(re.search(r'\d+', self.instance_dict[node.text]).group(0)))
                self.mydict[node.text] = num[len(num)-1]
            else:
                if(node.text in self.varMap):
                    pass
                else:
                    self.varMap[node.text] = self.count
                    self.count += 1
    
    def visit_operator(self, node, children):
        if '!=' in node.text:
            self.currentOperator = 'not(= '
        elif '==' in node.text:
            self.currentOperator = '= '
        elif '=<' in node.text:
            self.currentOperator = '<='
        else:
            self.currentOperator = node.text

    def visit_arith_op1(self, node, children):
        self.current_arith_operator1 = node.text

    def visit_arith_op2(self, node, children):
        self.current_arith_operator2 = node.text
    
    def visit_number(self, node, children):
        self.feVal = float(node.text)

    def visit_const(self, node, children):
        self.const = True

    def visit_expr1(self, node, children):
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        if(self.mydict['no_mapping'] == 'True'):
            assertStmnt = ('(assert(not (', self.currentOperator,' Class', str(0), ' ', str(self.feVal), ')))')
        else:    
            assertStmnt = ('(assert(not (', self.currentOperator,self.currentClass[0], ' ', str(self.feVal), ')))')
        f = open('assertStmnt.txt', 'a')
        for x in assertStmnt:
            f.write(x)
        if(self.currentOperator == 'not(= '):
            f.write(')')
        f.close()    
        
    def checkModelName(self):
        if(self.modelVarList[0] != self.modelVarList[1]):
            raise Exception('Model names do not match')
            sys.exit(1)
    
    def visit_expr2(self, node, children):
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        self.checkFeConsist()
        self.checkModelName()
        assertStmnt = ('(assert(not (', self.currentOperator,self.currentClass[0], ' ', self.currentClass[1], ')))')
        f = open('assertStmnt.txt', 'a')
        f.write('\n')
        for x in assertStmnt:
            f.write(x)
        if self.currentOperator == 'not(= ':
            f.write(')')
        f.close() 
        
    def visit_expr3(self, node, children):
        if self.count > int(self.paramDict['no_of_params']):
            raise Exception('The no. of parameters mentioned exceeded in assert statement')
            sys.exit(1)
        self.checkModelName()
        if self.negOp == '~':
            if self.paramDict['white_box_model'] == 'DNN':
                assertStmnt = ('(assert(not (', self.currentOperator,' (= ', self.classNameList[0],str(self.count-1),
                               ' 1)', ' (not ', ' (= ', self.classNameList[1],str(self.count-1),' 1)','))))')
            else:
                assertStmnt = ('(assert(not (', self.currentOperator,' (= ', self.classNameList[0],' 1)', 
                           ' (not ', ' (= ', self.classNameList[1],' 1)','))))')
        else:
            if self.paramDict['white_box_model'] == 'DNN':
                assertStmnt = ('(assert(not (', self.currentOperator,' (= ', self.classNameList[0],str(self.count-1),' 1)', ' ', 
                                                                ' (= ', self.classNameList[1],str(self.count-1),' 1)', ')))')
            else:    
                assertStmnt = ('(assert(not (', self.currentOperator,' (= ', self.classNameList[0],' 1)', ' ', 
                                                                ' (= ', self.classNameList[1],' 1)', ')))')
        f = open('assertStmnt.txt', 'a')
        f.write('\n')
        for x in assertStmnt:
            f.write(x)
        f.close()

    def visit_expr4(self, node, children):
        f = open('assertStmnt.txt', 'a')
        f.write('(define-fun min2 ((x1 Real) (x2 Real)) Real \n')
        f.write('    (ite (<= x1 x2) x1 x2))\n')
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()

        for i in range(1, len(self.fename_type)):
            if i >= 3:
                f.write('(define-fun min' + str(i) + ' (')
                for j in range(0, i):
                    f.write('(x' + str(j + 1) + ' Real)')
                f.write(') Real \n')
                f.write('   (min2 x1 (min' + str(i - 1) + ' ')
                for j in range(1, i):
                    f.write('x' + str(j + 1) + ' ')
                f.write(')))\n \n')
        f.write('(assert (not ('+self.currentOperator+' Class0 (min' + str(len(self.fename_type) - 1) + ' ')
        fe_list = list(self.fename_type.keys())
        for i in range(0, len(self.fename_type) - 1):
            f.write(fe_list[i] + '0' + ' ')
        f.write('))))\n')
        f.close()

    def visit_expr5(self, node, children):
        f = open('assertStmnt.txt', 'a')
        f.write('(define-fun max2 ((x1 Real) (x2 Real)) Real \n')
        f.write('    (ite (> x1 x2) x1 x2))\n')
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()

        for i in range(1, len(self.fename_type)):
            if i >= 3:
                f.write('(define-fun max' + str(i) + ' (')
                for j in range(0, i):
                    f.write('(x' + str(j + 1) + ' Real)')
                f.write(') Real \n')
                f.write('   (max2 x1 (max' + str(i - 1) + ' ')
                for j in range(1, i):
                    f.write('x' + str(j + 1) + ' ')
                f.write(')))\n \n')
        f.write('(assert (not ('+self.currentOperator+' Class0 (max' + str(len(self.fename_type) - 1) + ' ')
        fe_list = list(self.fename_type.keys())
        for i in range(0, len(self.fename_type) - 1):
            f.write(fe_list[i] + '0' + ' ')
        f.write('))))\n')
        f.close()

    #Lipschitz
    def visit_expr6(self, node, children):
        f = open('assertStmnt.txt', 'a')
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        if self.const:
            f.write('(assert (and (>= constnt 0.0) (<= constnt 1.0)))\n')
        f.write('(assert (= temp (+ ')
        fe_list = list(self.fename_type.keys())
        for i in range(0, len(self.fename_type) - 1):
            f.write('(absoluteReal (- '+fe_list[i] + '0 ' + fe_list[i] +'1))')
        f.write(')))\n')
        if self.const:
            f.write(
                '(assert (not (' + self.currentOperator + ' (absoluteReal (' + self.current_arith_operator1 + ' ' + 'Class0' +
                ' ' + 'Class1' + ')) (' + self.current_arith_operator2 + ' constnt temp))))\n')
        else:
            f.write('(assert (not ('+self.currentOperator+ ' (absoluteReal ('+self.current_arith_operator1+' '+'Class0'+
                ' ' + 'Class1'+')) (' + self.current_arith_operator2+' '+str(self.feVal)+' temp))))\n')
        f.close()

    def visit_expr7(self, node, children):
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        f = open('assertStmnt.txt', 'a')
        fe_list = list(self.fename_type.keys())
        f.write('(assert (not (= Class0 ' + fe_list[0]+str(0) + ')))\n')
        '''
        f.write('(assert (and (not ')
        for i in range(0, len(self.fename_type) - 1):
            f.write('('+ self.currentOperator + 'Class0 '+fe_list[i]+'0)')
        f.write(')))\n')
        '''
        f.close()

    def visit_expr8(self, node, children):
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        f = open('assertStmnt.txt', 'a')
        fe_list = list(self.fename_type.keys())
        f.write('(assert (and (= '+fe_list[0]+'1 '+fe_list[1]+'0)'+'(= '+fe_list[1]+'1 '+fe_list[0]+'0)')
        for i in range(2, len(self.fename_type) - 1):
            f.write('(= '+fe_list[i]+'1 '+fe_list[i]+'0)')
        f.write('))\n')
        f.write('(assert (not (= Class0 Class1)))')
        f.close()

    def visit_expr9(self, node, children):
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        f = open('assertStmnt.txt', 'a')
        fe_list = list(self.fename_type.keys())
        #f.write('(assert (and (= ' + fe_list[0] + '1 ' + fe_list[len(self.fename_type) - 2] + '0)')
        f.write('(assert (and ')
        for i in range(0, len(self.fename_type) - 2):
            f.write('(= ' + fe_list[i] + '1 ' + fe_list[i + 1] + '0)')
        f.write('))\n')
        f.write('(assert (and (= ' + fe_list[len(self.fename_type) - 2] + '1 ' + fe_list[0] + '0)))\n')
        f.write('(assert (not (= Class0 Class1)))\n')
        f.close()


    def visit_expr10(self, node, children):
        f = open('assertStmnt.txt', 'a')
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        f.write('(define-fun max2 ((x1 Real) (x2 Real)) Real \n')
        f.write('    (ite (> x1 x2) x1 x2))\n')
        f.write('(define-fun min2 ((x1 Real) (x2 Real)) Real \n')
        f.write('    (ite (<= x1 x2) x1 x2))\n')

        for i in range(1, len(self.fename_type)):
            if i >= 3:
                f.write('(define-fun min' + str(i) + ' (')
                for j in range(0, i):
                    f.write('(x' + str(j + 1) + ' Real)')
                f.write(') Real \n')
                f.write('   (min2 x1 (min' + str(i - 1) + ' ')
                for j in range(1, i):
                    f.write('x' + str(j + 1) + ' ')
                f.write(')))\n \n')

        for i in range(1, len(self.fename_type)):
            if i >= 3:
                f.write('(define-fun max' + str(i) + ' (')
                for j in range(0, i):
                    f.write('(x' + str(j + 1) + ' Real)')
                f.write(') Real \n')
                f.write('   (max2 x1 (max' + str(i - 1) + ' ')
                for j in range(1, i):
                    f.write('x' + str(j + 1) + ' ')
                f.write(')))\n \n')
                
        f.write('(assert (not (and ')
        f.write('(>= Class0 (min' + str(len(self.fename_type) - 1) + ' ')
        fe_list = list(self.fename_type.keys())
        for i in range(0, len(self.fename_type) - 1):
            f.write(fe_list[i] + '0' + ' ')
        f.write('))\n')

        f.write('(<= Class0 (max' + str(len(self.fename_type) - 1) + ' ')
        fe_list = list(self.fename_type.keys())
        for i in range(0, len(self.fename_type) - 1):
            f.write(fe_list[i] + '0' + ' ')
        f.write('))\n')
        f.write(')))\n')
        f.close()

    def visit_expr11(self, node, children):
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        f = open('assertStmnt.txt', 'a')
        #f.write('(assert ('+self.currentOperator+' Class0 const))\n')

        f.write('(assert (or ')
        fe_list = list(self.fename_type.keys())
        for i in range(0, len(self.fename_type) - 1):
            f.write('('+self.currentOperator+'Class0 '+fe_list[i]+'0)')

        f.write('))\n')
        f.close()

    def visit_expr12(self, node, children):
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()
        f = open('assertStmnt.txt', 'a')
        fe_list = list(self.fename_type.keys())
        '''
        for i in range(0, len(self.fename_type)):
            #if 'int' in fe_type:
            #    f.write("(declare-fun " + fe_list[i] + "2 () Int)\n")
            #elif ('float' in fe_type):
            f.write("(declare-fun " + fe_list[i] + "2 () Real)\n")
        '''
        f.write('(assert (and ')
        for i in range(0, len(self.fename_type) - 1):
            f.write('(= '+fe_list[i]+'2 (+ '+fe_list[i]+'0 '+fe_list[i]+'1))')
        f.write('))\n')
        f.write('(assert (not (= Class2 (+ Class0 Class1))))\n')
        f.close()


    def visit_expr13(self, node, children):
        f1 = open('logicAssert.txt', 'w')
        f1.write(node.text)
        f1.close()

        self.checkFeConsist()
        self.checkModelName()
        assertStmnt = ('(assert(not (', self.currentOperator, self.currentClass[1], ' (',
                       self.current_arith_operator1,' ',self.currentClass[0], ' ',str(self.feVal), '))))')
        f = open('assertStmnt.txt', 'a')
        f.write('\n')
        for x in assertStmnt:
            f.write(x)
        if self.currentOperator == 'not(= ':
            f.write(')')
        f.close()



    def checkFeConsist(self):


        if len(self.varList) == len(self.mydict)-2:
            for el in self.varList:
                if el not in self.mydict.keys():
                    raise Exception("Unknown feature vector")
                    sys.exit(1)
        else:
            raise Exception("No. of feature vectors do not match with the assumption")
            sys.exit(1)
