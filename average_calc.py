from utils.mlCheck import Assume, Assert, propCheck
import statistics
import csv as cv
import pandas as pd
import numpy as np
import graphviz
from sklearn import tree
from utils import tree2Logic
from joblib import load
import random
from scipy.stats import gmean
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import os



def funcWriteXml(df):
    f = open('dataInput.xml', 'w')
    f.write('<?xml version="1.0" encoding="UTF-8"?> \n <Inputs> \n')

    for i in range(0, df.shape[1]):
        f.write('<Input> \n <Feature-name>')
        f.write(df.columns.values[i])
        f.write('<\Feature-name> \n <Feature-type>')
        f.write(str(df.dtypes[i]))
        f.write('<\Feature-type> \n <Value> \n <minVal>')
        f.write(str(format(df.iloc[:, i].min(), '.7f')))
        f.write('<\minVal> \n <maxVal>')
        f.write(str(format(df.iloc[:, i].max(), '.7f')))
        f.write('<\maxVal> \n <\Value> \n <\Input>\n')

    f.write('<\Inputs>')
    f.close()


def func_init(ml):
    df = pd.read_csv('Mean_training.csv')
    #funcWriteXml(df)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    model = DecisionTreeRegressor()
    model.fit(X, y)

    propCheck(max_samples=1000, model_type='sklearn', model_path='Fairness_models/'+ml, instance_list = ['x', 'y'],
              xml_file='dataInput.xml', white_box_model='DNN', regression='no', bound_cex=True, bound_list=['Class'],
              bound_all_features=True, no_of_class=1, train_data_available=True, train_data_loc= 'Fairness_Datasets/bank.csv',train_ratio=10,
              layer_size=[3],  no_EPOCHS=50, no_of_train=500, mul_cex=True, solver = 'z3')

    # Monotonicity
    for i in range(0, 15):
        if i == 0:
            Assume('x[i] != y[i]', i)
        else:
            #Assume('0.01*abs(x[i] - y[i]) <= 0.1', i)
            Assume('x[i] = y[i]', i)
    Assert('model.predict(x) == model.predict(y)')


   


if __name__ == "__main__":
    model = ['DecisionTreeTitanic', 'GBTitanic', 'NBTitanic', 'RandomForestTitanic', 'LogRegTitanic', 'MLPTitanic']
    model_bank = ['DecisionTreeBank', 'GBBank', 'NBBank', 'RandomForestBank', 'LogRegBank', 'MLPBank']
    model_adult = ['DecisionTreeAdult', 'GBAdult', 'NBAdult', 'RandomForestAdult', 'LogRegAdult', 'MLPAdult']
    model_credit = ['DecisionTreeCredit', 'GBCredit', 'NBCredit', 'RandomForestCredit', 'LogRegCredit', 'MLPCredit']
    #model_credit = ['LogRegCredit', 'MLPCredit']
    for ml in model_bank:
        f = open('results_award', 'a')
        f.write('\n-------------'+ml+'--------------------\n')
        f.close()
        for i in range(0, 5):
            func_init(ml)
    

