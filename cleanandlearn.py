import pandas as pd
import numpy as np 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import imageio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def keepBest(data):
    alldata = np.unique(data[['dimension', 'budget', 'lambda']], axis=0)
    print(alldata.shape)
    cleaned_data = pd.DataFrame(columns=['dimension', 'budget', 'lambda', 'mu'])
    for a in alldata:
        d = a[0]
        b = a[1]
        l = a[2]
        tmp = data[data['dimension'] == d]
        tmp = tmp[tmp['budget'] == b]
        tmp = tmp[tmp['lambda'] == l]
        themin = tmp[tmp.loss == tmp.loss.min()]
        cleaned_data = cleaned_data.append(themin)
    del(cleaned_data['loss'])
    cleaned_data = cleaned_data.astype(dtype=float)
    print(cleaned_data.shape)
    return cleaned_data

def analyze(data):
    print(data)
    print(data.shape)
    print(data.info())
    print(data.describe())
    print(data.head())


def splitData(data, test_ratio):
    train, test = train_test_split(data, test_size=test_ratio)
    x_train = train
    y_train = train['mu']
    del(train['mu'])
    x_test = test
    y_test = test['mu']
    del(test['mu'])
    return x_train, y_train, x_test, y_test


def create_model(classifier, x, y):
    classifier.fit(x, y)
    return classifier


def display_score(classifier, x_train, y_train, x_test, y_test):
    y_pred = classifier.predict(x_test)
    print('Coefficient of determination: %s' % r2_score(y_test, y_pred))
    print('MAE: %s' % mean_absolute_error(y_test, y_pred))
    print('MSE: %s' % mean_squared_error(y_test, y_pred))


###############################################
###### With pytorch
###############################################
class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 100
        
        self.hidden = nn.Linear(self.inputSize, self.hiddenSize)
        self.output = nn.Linear(self.hiddenSize, self.outputSize)
        
    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return x
               

def main():
    data = pd.read_csv('data.csv', sep=',')
    cleaned_data = keepBest(data)
    # analyze(cleaned_data)
    x_train, y_train, x_test, y_test = splitData(cleaned_data, 0.1)
    print(x_train.shape)
    print(x_test.shape)

    model = Model()
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)



    data_train = torch.utils.data.TensorDataset(torch.Tensor(np.array(x_train)), torch.Tensor(np.array(y_train)))
    train_loader = torch.utils.data.DataLoader(data_train, batch_size = 16, shuffle = True)

    data_test = torch.utils.data.TensorDataset(torch.Tensor(np.array(x_test)), torch.Tensor(np.array(y_test)))
    test_loader = torch.utils.data.DataLoader(data_test)

    for epoch in range(1, 1001): ## run the model for 10 epochs
        train_loss, valid_loss = [], []

        ## training part 
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()

            ## 1. forward propagation
            output = model(data)

            ## 2. loss calculation
            loss = loss_function(output, target.view(-1,1))

            ## 3. backward propagation
            loss.backward()

            ## 4. weight optimization
            optimizer.step()

            train_loss.append(loss.item())

        ## evaluation part 
        model.eval()
        for data, target in test_loader:
            output = model(data)
            loss = loss_function(output, target.view(-1,1))
            valid_loss.append(loss.item())
            print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))

    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))


if __name__ == '__main__':
    main()
