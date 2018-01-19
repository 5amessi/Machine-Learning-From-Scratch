import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class SoftmaxRegression(object):
    def __init__(self, learning_rate=0.01, epochs=50):
        self.__epochs= epochs
        self.__learning_rate = learning_rate
        self.OneHotEn = {}

    def fit(self, X, Y):
        classes = list(set(Y))
        self.One_Hot(Y)
        Y = self.encoding(Y)
        self.w_ = np.zeros((X.shape[1], len(classes)))
        self.b = np.ones((1,len(classes)))
        self.cost_ = []

        for i in range(self.__epochs):
            y_ = self.__net_input(X, self.w_, self.b)
            activated_y = self.__activation(y_)
            errors = (Y - activated_y)
            self.w_ += self.__learning_rate * X.T.dot(errors)
            self.b += self.__learning_rate * errors.sum()
            self.cost_.append(self.__cost(self._cross_entropy(output=activated_y, y_target=Y)))

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * (y_target), axis=1)

    def __cost(self, cross_entropy):
        return 0.5 * np.mean(cross_entropy)

    def __softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def __net_input(self, X, W, b):
        return (X.dot(W) + b)

    def __activation(self, X):
        return self.__softmax(X)

    def predict(self, X):
        z = self.__net_input(X, self.w_, self.b)
        activated_z = self.__softmax(z)
        max_indices = np.argmax(activated_z,axis=1)+1
        return max_indices

    def One_Hot(self,input):
        data = list(set(input))
        arr = np.zeros((len(data),len(data)))
        for i in range (len(data)):
            arr[i][i] = 1
            self.OneHotEn[data[i]] = arr[i]

    def encoding(self,train_y):
        lables = []
        for i in range(len(train_y)):
            lables.append(self.OneHotEn[train_y[i]])
        return lables

    def plot(self):
        plt.plot(range(1, len(self.cost_) + 1), np.log10(self.cost_))
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Softmax Regression - Learning rate 0.02')
        plt.tight_layout()
        plt.show()
