import pandas as pd
import numpy as np 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def keepBest(data):
    alldata = np.unique(data[['dimension', 'budget', 'lambda']], axis=0)
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


def main():
    data = pd.read_csv('data.csv', sep=',')
    # print(data)
    cleaned_data = keepBest(data)
    analyze(cleaned_data)
    x_train, y_train, x_test, y_test = splitData(cleaned_data, 0.3)
    print(x_test)
    print(y_test)
    classifier = create_model(MLPRegressor(hidden_layer_sizes=(10,2)), x_train, y_train)
    display_score(classifier, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
