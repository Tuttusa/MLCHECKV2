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
        self.min = ""
        self.max = ""
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

    def generic_visit(self, node, children):
        pass

    def visit_operator(self, node, children):
        if '!=' in node.text:
            self.currentOperator = 'not(= '
        elif '=' == node.text:
            self.currentOperator = '=='
        elif '=<' in node.text:
            self.currentOperator = '<='
        else:
            self.currentOperator = node.text

    def visit_min_symbol(self, node, children):
        self.min_max = node.text

    def visit_max_symbol(self, node, children):
        self.min_max = node.text

    def visit_arith_op1(self, node, children):
        self.current_arith_operator1 = node.text

    def visit_arith_op2(self, node, children):
        self.current_arith_operator2 = node.text

    def visit_const(self, node, children):
        self.const = True

    def visit_number(self, node, children):
        self.feVal = float(node.text)

    def write_pre_lines(self, f, dist_flag):
        f.write('import mlCheck\n')
        f.write('import pandas as pd\n')
        f.write('import csv as cv\n')
        f.write('import numpy as np\n')
        f.write('from utils import util, mlCheck\n')
        f.write('from operator import add\n')
        if dist_flag:
            f.write('def manhattan_distance(a, b):\n')
            f.write('    return np.abs(a - b).sum()\n')
        f.write('def func_match_mut_pred(X, model, arr_length):\n')
        f.write('    obj_mlcheck = mlCheck.runChecker()\n')
        f.write('    retrain_flag = False\n')
        f.write('    with open(\'param_dict.csv\') as csv_file:\n')
        f.write('        reader = cv.reader(csv_file)\n')
        f.write('        paramDict = dict(reader)\n')
        f.write('    mul_cex = paramDict[\'mul_cex_opt\']\n')
        f.write('    no_of_params = int(paramDict[\'no_of_params\'])\n')
        f.write('    testIndx = 0\n')
        f.write('    while testIndx < arr_length:\n')
        f.write('        temp_store = []\n')
        f.write('        temp_add_oracle = []\n')

    def write_post_lines(self, f):
        f.write('        if not retrain_flag:\n')
        f.write('            if mul_cex == \'True\':\n')
        f.write('                with open(\'CexSet.csv\', \'a\', newline=\'\') as csvfile:\n')
        f.write('                    writer = cv.writer(csvfile)\n')
        f.write('                    writer.writerows(temp_store)\n')
        f.write('            else:\n')
        f.write('                print(\'A counter example is found, check it in CexSet.csv file: \', temp_store)\n')
        f.write('                with open(\'CexSet.csv\', \'a\', newline=\'\') as csvfile:\n')
        f.write('                    writer = cv.writer(csvfile)\n')
        f.write('                    writer.writerows(temp_store)\n')
        f.write('                obj_mlcheck.addModelPred()\n')
        f.write('                return 1\n')
        f.write('        else: \n')
        f.write('            util.funcAdd2Oracle(temp_add_oracle)\n')
        f.write('            obj_mlcheck.funcCreateOracle()\n')
        f.write('    return 0\n')
        f.close()

    # Annihilator
    def visit_expr1(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        f.write('        if not(model.predict(np.reshape(X[testIndx], (1, -1))) ' + self.currentOperator + ' ' + str(
            self.feVal) + '):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        self.write_post_lines(f)

    # Monotonicity
    def visit_expr2(self, node, children):
        f = open('utils/match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        print(self.currentOperator)
        f.write('        if not(model.predict(np.reshape(X[testIndx], (1, -1)))[0] ' + self.currentOperator +
                ' model.predict(np.reshape(X[testIndx+1], (1, -1)))[0]):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            temp_store.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            temp_add_oracle.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        self.write_post_lines(f)

    # Conjunctivity
    def visit_expr4(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        f.write('        if not(model.predict(np.reshape(X[testIndx], (1, -1))) <= min(X[testIndx])):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        self.write_post_lines(f)

    # Disjunctivity
    def visit_expr5(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        f.write('        if not(model.predict(np.reshape(X[testIndx], (1, -1))) >= max(X[testIndx])):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        self.write_post_lines(f)

    # Internality
    def visit_expr10(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        f.write(
            '        if not(min(X[testIndx]) <= model.predict(np.reshape(X[testIndx], (1, -1))) <= max(X[testIndx])):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        self.write_post_lines(f)

    # Idempotency/Annihilator
    def visit_expr7(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        f.write('        if not(np.round(model.predict(np.reshape(X[testIndx], (1, -1))), 10)' + self.currentOperator +
                ' ' + 'np.round(X[testIndx][0], 10)):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        self.write_post_lines(f)

    # Symmetric condition 1
    def visit_expr8(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        f.write('        if not(np.round(model.predict(np.reshape(X[testIndx], (1, -1))), 10) == '
                'np.round(model.predict(np.reshape(X[testIndx+1], (1, -1))), 10)):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            temp_store.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            temp_add_oracle.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        self.write_post_lines(f)

    # Symmetric condition 2
    def visit_expr9(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        f.write('        if not(np.round(model.predict(np.reshape(X[testIndx], (1, -1))), 10) == '
                'np.round(model.predict(np.reshape(X[testIndx+1], (1, -1))), 10)):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            temp_store.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            temp_add_oracle.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        self.write_post_lines(f)

    # Anihilator
    def visit_expr11(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        f.write('        for i in range(0, X.shape[1]):\n')
        f.write(
            '           if model.predict(np.reshape(X[testIndx], (1, -1)))' + self.currentOperator + ' ' + 'X[testIndx][i]:\n')
        f.write('               retrain_flag = False\n')
        f.write('               temp_store.append(X[testIndx])\n')
        f.write('               testIndx += 1\n')
        f.write('               break\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            testIndx += 1\n')
        self.write_post_lines(f)

    # Additivity
    def visit_expr12(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)

        f.write('        temp_list = list(map (add, X[testIndx], X[testIndx+1]))\n')
        f.write('        if not(np.round(model.predict(np.reshape(X[testIndx], (1, -1)))[0] + '
                'model.predict(np.reshape(X[testIndx+1], (1, -1)))[0], 12) == '
                'np.round(model.predict(np.reshape(temp_list, (1, -1))), 12)):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            temp_store.append(X[testIndx+1])\n')
        f.write('            temp_store.append(temp_list)\n')
        f.write('            testIndx += 3\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            temp_add_oracle.append(X[testIndx+1])\n')
        f.write('            temp_add_oracle.append(temp_list)\n')
        f.write('            testIndx += 3\n')
        self.write_post_lines(f)

    def visit_expr13(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, False)
        #f.write('        print(\'testIndex---------\',testIndx)\n') 
        #f.write('        print(np.round(' + str(self.feVal) + self.current_arith_operator1 +
        #                        '(model.predict(np.reshape(X[testIndx], (1, -1)))[0]), 10))\n')
        
        #f.write('        print(np.round(model.predict(np.reshape(X[testIndx+1], (1, -1)))[0], 10))\n')
        f.write('        if not(np.round(' + str(self.feVal) + self.current_arith_operator1 +
                'model.predict(np.reshape(X[testIndx], (1, -1)))[0], 10) ' + self.currentOperator +
                ' np.around(model.predict(np.reshape(X[testIndx+1], (1, -1)))[0], 10)):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            temp_store.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            temp_add_oracle.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        self.write_post_lines(f)

    # Lipschitz
    def visit_expr6(self, node, children):
        f = open('match_mutprediction.py', 'w')
        self.write_pre_lines(f, True)
        f.write('        if not(np.round(abs(model.predict(np.reshape(X[testIndx], (1, -1)))[0] - '
                'model.predict(np.reshape(X[testIndx+1], (1, -1)))[0]), 10) <= '
                'np.round(' + str(self.feVal) + '*manhattan_distance(X[testIndx], X[testIndx+1]), 10)):\n')
        f.write('            retrain_flag = False\n')
        f.write('            temp_store.append(X[testIndx])\n')
        f.write('            temp_store.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        f.write('        else:\n')
        f.write('            retrain_flag = True\n')
        f.write('            temp_add_oracle.append(X[testIndx])\n')
        f.write('            temp_add_oracle.append(X[testIndx+1])\n')
        f.write('            testIndx += 2\n')
        self.write_post_lines(f)

    def checkModelName(self):
        if (self.modelVarList[0] != self.modelVarList[1]):
            raise Exception('Model names do not match')
            sys.exit(1)


def assert_rev(*args):
    grammar = Grammar(
        r"""
        expr        = expr13/ expr1 / expr2/ expr3/ expr4/ expr5 / expr6/ expr7/ expr8 /expr9/ expr10/ expr11 / expr12
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
    assert_visit = AssertionVisitor()
    assert_visit.visit(tree)
