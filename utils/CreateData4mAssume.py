import pandas as pd
import csv as cv
import numpy as np
from parsimonious.nodes import NodeVisitor
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
import random as rd
import re


class AssumptionVisitor(NodeVisitor):
    def __init__(self):
        self.logic_operator = ''
        self.featureList = []
        self.num = ''
        self.index = -1
        with open('param_list.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.paramList = dict(reader)
        with open('feNameType.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.fename_type = dict(reader)
        self.final_dataset = np.zeros((len(list(self.paramList)) * 5, len(list(self.fename_type))))
        #indexes = np.zeros((1, len(list(self.paramList)) * 1000))
        indexes = []
        for i in range(0, len(list(self.paramList)) * 10):
            indexes.append(list(self.paramList.values())[i % len(list(self.paramList))])
        self.df_value_range = pd.read_csv('FeatureValueRange.csv')
        #df = pd.read_csv('AssumeOracle.csv')
        #df.insert(0, 'index', indexes)
        #df.to_csv('AssumeOracle.csv', index=False, header=True)

    def generic_visit(self, node, children):
        pass

    def visit_logic_op(self, node, children):
        self.logic_operator = node.text

    def visit_var(self, node, children):
        self.featureList.append(node.text)

    def visit_number(self, node, children):
        self.num = node.text

    def getOperator(self, bound, index, temp_index, param_no):
        if self.logic_operator.strip() == '<=':
            if 'float' in self.fe_type:
                self.final_dataset[param_no[index]][temp_index] = rd.uniform(bound, self.df_value_range.iloc[1][temp_index])
            else:
                self.final_dataset[param_no[index]][temp_index] = rd.randint(bound, self.df_value_range.iloc[1][temp_index])
        elif self.logic_operator.strip() == '<':
            if 'float' in self.fe_type:
                self.final_dataset[param_no[index]][temp_index] = rd.uniform(bound + 1, self.df_value_range.iloc[1][temp_index])
            else:
                self.final_dataset[param_no[index]][temp_index] = rd.randint(bound + 1, self.df_value_range.iloc[1][temp_index])
        elif self.logic_operator.strip() == '>=':
            if 'float' in self.fe_type:
                self.final_dataset[param_no[index]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], bound)
            else:
                self.final_dataset[param_no[index]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], bound)
        elif self.logic_operator.strip() == '>':
            if 'float' in self.fe_type:
                self.final_dataset[param_no[index]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index],bound - 0.00001)
            else:
                self.final_dataset[param_no[index]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], bound - 1)
        elif self.logic_operator.strip() == '=':
            self.final_dataset[param_no[index]][temp_index] = bound
        elif self.logic_operator.strip() == '!=':
            while True:
                if 'float' in self.fe_type:
                    self.final_dataset[param_no[index]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index],
                                                                                 self.df_value_range.iloc[1][temp_index])
                else:
                    self.final_dataset[param_no[index]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                if self.final_dataset[param_no[index]][temp_index] != bound: break

    def write_data(self, *args):
        df = pd.read_csv('AssumeOracle.csv')

        if len(args) == 2:
            #for i in range(0, self.final_dataset.shape[1]):
            df.iloc[args[0]][self.index] = self.final_dataset[args[0]][self.index]
            df.iloc[args[0]]['Class'] += 1
            #for i in range(0, self.final_dataset.shape[1]):
            df.iloc[args[1]][self.index] = self.final_dataset[args[1]][self.index]
            df.iloc[args[1]]['Class'] += 1
        else:
            #for i in range(0, self.final_dataset.shape[1]):
            df.iloc[args[0]][self.index] = self.final_dataset[args[0]][self.index]
            df.iloc[args[0]]['Class'] += 1
        df.to_csv('AssumeOracle.csv', index=False, header=True)

    def visit_expr1(self, node, children):
        param_no = []
        temp_index = -1
        temp_dict = {}
        dataset_update_flag = False
        df = pd.read_csv('AssumeOracle.csv')
        with open('countInstance.txt') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        count_instance = int(content[0])
        for i in range(0, len(self.featureList)):
            for j in range(0, len(list(self.fename_type))):
                if list(self.fename_type)[j] in self.featureList[i]:
                    temp_index = j
                    self.index = temp_index
                    self.fe_type = self.fename_type[list(self.fename_type)[j]]
                    param_no.append(int(self.featureList[i].strip().rsplit(list(self.fename_type)[j], 1)[1])+count_instance*len(list(self.paramList)))
                    break
        if len(param_no) == 2:
            if df.iloc[param_no[0]]['Class'] != 0:
                self.final_dataset[param_no[0]][temp_index] = df.iloc[param_no[0]][temp_index]
                self.getOperator(self.final_dataset[param_no[0]][temp_index], 1, temp_index, param_no)
                self.write_data(param_no[0], param_no[1])
            elif df.iloc[param_no[1]]['Class'] != 0:
                self.final_dataset[param_no[1]][temp_index] = df.iloc[param_no[1]][temp_index]
                if self.logic_operator.strip() == '<=':
                    if 'float' in self.fe_type:
                        self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], self.final_dataset[param_no[1]][
                            temp_index])
                    else:
                        self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], self.final_dataset[param_no[1]][temp_index])
                elif self.logic_operator.strip() == '<':
                    if 'float' in self.fe_type:
                        self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], self.final_dataset[param_no[1]][
                            temp_index] - 0.00001)
                    else:
                        self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], self.final_dataset[param_no[1]][temp_index]-1)
                elif self.logic_operator.strip() == '>=':
                    if 'float' in self.fe_type:
                        self.final_dataset[param_no[0]][temp_index] = rd.uniform(
                            self.final_dataset[param_no[1]][temp_index], self.df_value_range.iloc[1][temp_index])
                    else:
                        self.final_dataset[param_no[0]][temp_index] = rd.randint(self.final_dataset[param_no[1]][temp_index], self.df_value_range.iloc[1][temp_index])
                elif self.logic_operator.strip() == '>':
                    if 'float' in self.fe_type:
                        self.final_dataset[param_no[0]][temp_index] = rd.uniform(
                            self.final_dataset[param_no[1]][temp_index] + 0.00001, self.df_value_range.iloc[1][temp_index])
                    else:
                        self.final_dataset[param_no[0]][temp_index] = rd.randint(self.final_dataset[param_no[1]][temp_index]+1, self.df_value_range.iloc[1][temp_index])
                elif self.logic_operator.strip() == '=':
                    self.final_dataset[param_no[0]][temp_index] = self.final_dataset[param_no[1]][temp_index]
                elif self.logic_operator.strip() == '!=':
                    while True:
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                        if self.final_dataset[param_no[0]][temp_index] != self.final_dataset[param_no[1]][temp_index]: break
                self.write_data(param_no[0], param_no[1])
            else:
                if 'float' in self.fe_type:
                    self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                else:
                    self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                self.getOperator(self.final_dataset[param_no[0]][temp_index], 1, temp_index, param_no)
                self.write_data(param_no[0], param_no[1])
        elif len(param_no) == 1:
            if node.text.index(self.num) < node.text.index(self.featureList[0]):
                if df.iloc[param_no[0]]['Class'] != 0:
                    if self.logic_operator.strip() == '<=' and not df.iloc[param_no[0]][temp_index] >= float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(float(self.num.strip()), self.df_value_range.iloc[1][temp_index])
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(int(self.num.strip()), self.df_value_range.iloc[1][temp_index])
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '<' and not df.iloc[param_no[0]][temp_index] > float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(float(self.num.strip()) + 0.00001, self.df_value_range.iloc[1][temp_index])
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(int(self.num.strip())+1, self.df_value_range.iloc[1][temp_index])
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '>=' and not df.iloc[param_no[0]][temp_index] <= float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], float(self.num.strip()))
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], int(self.num.strip()))
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '>' and not df.iloc[param_no[0]][temp_index] < float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], float(self.num.strip())-0.00001)
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], int(self.num.strip()) - 1)
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '=' and not df.iloc[param_no[0]][temp_index] == float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = float(self.num.strip())
                        else:
                            self.final_dataset[param_no[0]][temp_index] = int(self.num.strip())
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '!=' and not df.iloc[param_no[0]][temp_index] != float(self.num.strip()):
                        while True:
                            if 'float' in self.fe_type:
                                self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                            else:
                                self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                            if self.final_dataset[param_no[0]][temp_index] != float(self.num.strip()): break
                        self.write_data(param_no[0])
                else:
                    self.getOperator(int(self.num.strip()), 0, temp_index, param_no)
                    self.write_data(param_no[0])
            else:

                if df.iloc[param_no[0]]['Class'] != 0:
                    if self.logic_operator.strip() == '<=' and not df.iloc[param_no[0]][temp_index] <= float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], float(self.num.strip()))
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], int(self.num.strip()))
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '<' and not df.iloc[param_no[0]][temp_index] < float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], float(self.num.strip()) - 0.00001)
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], int(self.num.strip())-1)
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '>=' and not df.iloc[param_no[0]][temp_index] >= float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(float(self.num.strip()), self.df_value_range.iloc[1][temp_index])
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(int(self.num.strip()), 1000)
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '>' and not df.iloc[param_no[0]][temp_index] > float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(float(self.num.strip()) + 0.00001, self.df_value_range.iloc[1][temp_index])
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(int(self.num.strip())+1, self.df_value_range.iloc[1][temp_index])
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '=' and not df.iloc[param_no[0]][temp_index] == float(self.num.strip()):
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = float(self.num.strip())
                        else:
                            self.final_dataset[param_no[0]][temp_index] = int(self.num.strip())
                        self.write_data(param_no[0])
                    elif self.logic_operator.strip() == '!=' and not df.iloc[param_no[0]][temp_index] != float(self.num.strip()):
                        while True:
                            if 'float' in self.fe_type:
                                self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                            else:
                                self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                            if self.final_dataset[param_no[0]][temp_index] != float(self.num.strip()): break
                        self.write_data(param_no[0])
                else:
                    if self.logic_operator.strip() == '<=':
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], float(self.num.strip()))
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], int(self.num.strip()))
                    elif self.logic_operator.strip() == '<':
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], float(self.num.strip()) - 0.00001)
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], int(self.num.strip())-1)
                    elif self.logic_operator.strip() == '>=':
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(float(self.num.strip()), self.df_value_range.iloc[1][temp_index])
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(int(self.num.strip()), self.df_value_range.iloc[1][temp_index])
                    elif self.logic_operator.strip() == '>':
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = rd.uniform(float(self.num.strip()) + 0.00001, self.df_value_range.iloc[1][temp_index])
                        else:
                            self.final_dataset[param_no[0]][temp_index] = rd.randint(int(self.num.strip())+1, self.df_value_range.iloc[1][temp_index])
                    elif self.logic_operator.strip() == '=':
                        if 'float' in self.fe_type:
                            self.final_dataset[param_no[0]][temp_index] = float(self.num.strip())
                        else:
                            self.final_dataset[param_no[0]][temp_index] = int(self.num.strip())
                    elif self.logic_operator.strip() == '!=':
                        while True:
                            if 'float' in self.fe_type:
                                self.final_dataset[param_no[0]][temp_index] = rd.uniform(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                            else:
                                self.final_dataset[param_no[0]][temp_index] = rd.randint(self.df_value_range.iloc[0][temp_index], self.df_value_range.iloc[1][temp_index])
                            if self.final_dataset[param_no[0]][temp_index] != float(self.num.strip()): break
                    self.write_data(param_no[0])



class dataset_create:
    def __init__(self):
        pass


    def data_grammar(self, expression):
        grammar = Grammar(
            r"""

        expr        = expr1
        expr1       = logic_op ws? (var/number) ws? (var/number)
        var         = ~"([a-zA-Z_][a-zA-Z0-9_]*)"
        logic_op    = ws (geq / leq / eq / neq / and / lt / gt) ws
        op_beg      = number arith_op
        op_end      = arith_op number
        arith_op    = ws (add/sub/div/mul) ws
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
        fe_num      = ~"[+-]?([0-9]*[.])?[0-9]+"
        number      = ~"[+-]?([0-9]*[.])?[0-9]+"
        """
        )
        tree = grammar.parse(expression)
        assumeVisitObj = AssumptionVisitor()
        assumeVisitObj.visit(tree)

    def create_data(self):
        with open('assumeData.txt') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for i in range(0, len(content)):
            if 'not(=' in content[i]:
                temp_str = content[i]
                temp_str = temp_str.replace('not(=', '!=')
                content[i] = temp_str
        content = [s.replace('(','') for s in content]
        content = [s.replace(')','') for s in content]
        f.close()
        # how many data instances to be generated
        for j in range(0, 2):
            f = open('countInstance.txt', 'w')
            f.write(str(j))
            f.close()
            for i in range(len(content)-1, -1, -1):
                self.data_grammar(content[i])


