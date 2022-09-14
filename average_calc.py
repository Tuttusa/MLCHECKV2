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


def func_init():
    df = pd.read_csv('ecoli.csv')
    funcWriteXml(df)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    model = RandomForestRegressor()
    model.fit(X, y)

    propCheck(no_of_params=2, max_samples=1000, model_type='sklearn', model=model, param_list = ['x', 'y'],
              xml_file='dataInput.xml', white_box_model='Decision tree', regression='yes', bound_cex=True,
              no_of_class=1,
              layer_size=5, no_of_layers=2, no_EPOCHS=50, no_of_train=1000, upper_bound=1, lower_bound=0)

    # Monotonicity
    for i in range(0, 26):
        if i == 0:
            Assume('x[i] > y[i]', i)
        else:
            Assume('x[i] = y[i]', i)
    #Assert('model.predict(x) > model.predict(y)')


   


if __name__ == "__main__":
    func_init()
    

