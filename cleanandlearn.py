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
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


def accuracy(model, data_x, data_y, pct_close):
    n_items = len(data_y)
    X = torch.Tensor(data_x)  # 2-d Tensor
    Y = torch.Tensor(data_y)  # actual as 1-d Tensor
    oupt = model(X)       # all predicted as 2-d Tensor
    pred = oupt.view(n_items)  # all predicted as 1-d
    # print(X)
    print(Y)
    print(pred)
    n_correct = torch.sum((torch.abs(pred - Y) < torch.abs(pct_close * Y)))
    result = (n_correct.item() * 100.0 / n_items)  # scalar
    return result 


def learnPytorch(x_train, y_train, x_test, y_test):
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
    net = Net(n_feature=3, n_hidden=10, n_output=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()
    net = net.train()
    batch_size = 10
    n_items = len(x_train)
    batches_per_epoch = n_items // batch_size
    max_batches = 1000 * batches_per_epoch

    print("Starting training")
    for b in range(max_batches):
        curr_bat = np.random.choice(n_items, batch_size,
                                    replace=False)
        X = torch.Tensor(x_train[curr_bat])
        Y = torch.Tensor(y_train[curr_bat]).view(batch_size,1)
        # print(X)
        # print(Y)
        optimizer.zero_grad()
        oupt = net(X)
        loss_obj = loss_func(oupt, Y)
        loss_obj.backward()
        optimizer.step()
        if b % (max_batches // 10) == 0 or True:
            print("batch = %6d" % b, end="")
            print("  batch loss = %7.4f" % loss_obj.item(), end="")
            net = net.eval()
            acc = accuracy(net, x_train, y_train, 0.15)
            net = net.train()
            print("  accuracy = %0.2f%%" % acc)
            if b > 0:
                return      
    print("Training complete \n")

    # 4. Evaluate model
    net = net.eval()  # set eval mode
    acc = accuracy(net, x_test, y_test, 0.15)
    print("Accuracy on test data = %0.2f%%" % acc)

    # # 6. Use model
    # raw_inpt = np.array([[0.09266, 34, 6.09, 0, 0.433, 6.495, 18.4,
    #   5.4917, 7, 329, 16.1, 383.61, 8.67]], dtype=np.float32)
    # norm_inpt = np.array([[0.000970, 0.340000, 0.198148, -1,
    #   0.098765, 0.562177, 0.159629, 0.396666, 0.260870, 0.270992,
    #   0.372340, 0.966488, 0.191501]], dtype=np.float32)
    # X = T.Tensor(norm_inpt)
    # y = net(X)
    # print("For a town with raw input values: ")
    # for (idx,val) in enumerate(raw_inpt[0]):
    #   if idx % 5 == 0: print("")
    #   print("%11.6f " % val, end="")
    # print("\n\nPredicted median house price = $%0.2f" %
    #   (y.item()*10000)) 




    # # training
    # for t in range(200):
    
    #     prediction = net(x)     # input x and predict based on x
    #     loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    #     optimizer.zero_grad()   # clear gradients for next train
    #     loss.backward()         # backpropagation, compute gradients
    #     optimizer.step()        # apply gradients

    
class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 113
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 2 X 3 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
        
    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3) # final activation function
        return o
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self, xPredicted):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))


import torch.nn as nn
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
    # print(data)
    cleaned_data = keepBest(data)
    # analyze(cleaned_data)
    x_train, y_train, x_test, y_test = splitData(cleaned_data, 0.1)
    print(x_train.shape)
    print(x_test.shape)
    # learnPytorch(x_train, y_train, x_test, y_test)
    # x_train = x_train.to_numpy()
    # y_train = y_train.to_numpy()
    # x_test = x_test.to_numpy()
    # y_test = y_test.to_numpy()
    # X = torch.Tensor(x_train)
    # y = torch.Tensor(y_train)
    # print(X.shape)
    # print(y.shape)
    # NN = Neural_Network()
    # for i in range(1000):  # trains the NN 1,000 times
    #     print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    #     NN.train(X, y)
    # NN.saveWeights(NN)
    # NN.predict(x_test)
    # print(x_test)
    # print(y_test)

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



    # classifier = create_model(MLPRegressor(hidden_layer_sizes=(10,2)), x_train, y_train)
    # display_score(classifier, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
