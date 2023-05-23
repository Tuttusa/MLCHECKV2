

import torch
import torchvision
import numpy as np
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import csv as cv
from sklearn.neural_network import MLPRegressor, MLPClassifier
from joblib import dump



class LinearNet(nn.Module):
    def __init__(self, input_size):
        super(LinearNet, self).__init__()
        with open('param_dict.csv') as csv_file:
            reader = cv.reader(csv_file)
            self.mydict = dict(reader)   
        self.num_layers = int(self.mydict['no_of_layers'])
        self.layers_size = int(self.mydict['layer_size'])
        self.output_size = int(self.mydict['no_of_class'])
        self.linears = nn.ModuleList([nn.Linear(input_size, self.layers_size)])
        self.linears.extend([nn.Linear(self.layers_size, self.layers_size) for i in range(1, self.num_layers-1)])
        self.linears.append(nn.Linear(self.layers_size, self.output_size))
    
    def forward(self, x):
        for i in range(0, self.num_layers-1):
            x = F.relu(self.linears[i](x))
        x = self.linears[self.num_layers-1](x)
        if self.mydict['regression'] == 'yes':
            return x
        return F.log_softmax(x, dim=1)  


class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w = module.weight.data
            w = torch.clamp(w, min=-10, max=100)
            module.weight.data=w


def functrainDNN():
    df = pd.read_csv('OracleData.csv')
    data = df.values
    X = data[:,:-1]
    y = data[:, -1]

    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        mydict = dict(reader)   
    EPOCH = int(mydict['no_EPOCHS'])

    net = LinearNet(input_size=df.shape[1]-1)
    constraints = weightConstraint()
    if mydict['regression'] == 'yes':
        X_train = torch.from_numpy(X).float()
        y_train = torch.squeeze(torch.from_numpy(y).float())
        y_train = y_train.view(-1, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
        loss_func = torch.nn.MSELoss()
    else:
        X_train = torch.from_numpy(X).float()
        y_train = torch.squeeze(torch.from_numpy(y).long())
        loss_func = F.nll_loss()
        optimizer = optim.Adam(net.parameters(), lr =0.001)

    for epoch in range(0, EPOCH):
        optimizer.zero_grad()
        output = net(X_train)
        loss = loss_func(output, y_train)
        loss.backward()
        optimizer.step()
        for i in range(0, len(net.linears)):
            net.linears[i].apply(constraints)

    MODEL_PATH = 'Model/dnn_model'
    torch.save(net, MODEL_PATH)


def functrainDNNSklearn():
    df = pd.read_csv('OracleData.csv')
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]

    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        mydict = dict(reader)

    layer_size = eval(mydict['layer_size'])
    if mydict['regression'] == 'yes':
        dnn_model = MLPRegressor(hidden_layer_sizes=layer_size, max_iter=300).fit(X, y)

    else:
        dnn_model = MLPClassifier(hidden_layer_sizes=layer_size).fit(X, y)

    dump(dnn_model, 'Model/dnn_model_sklearn')