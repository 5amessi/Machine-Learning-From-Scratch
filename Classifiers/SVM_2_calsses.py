import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class svm():
    def __init__(self, learning_rate=0.01, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self,X, Y):
        self.weight = np.random.rand(len(X[0]))
        self.bias = 1
        for epoch in range(1, self.epochs):
            for i, raw_x in enumerate(X):
                if (Y[i] * (np.dot(raw_x, self.weight)+ self.bias)) > 1:
                    self.weight += self.learning_rate * (-2 * (1 / epoch) * self.weight)
                    self.bias   += self.learning_rate * (-2 * (1 / epoch) * 1)
                else:
                    self.weight += self.learning_rate * ((raw_x * Y[i]) + (-2 * (1 / epoch) * self.weight))
                    self.bias   += self.learning_rate *          (Y[i] + (-2 * (1 / epoch) * 1))

    def test(self,X):
        return np.sign(X.dot(self.weight)+self.bias)

    def fit(self, X, Y):
        self.unique_val, self.classes = np.unique(Y, return_counts=True)
        if len(self.classes) == 2:
            tempy = np.copy(Y)
            tempy[tempy == self.unique_val[0]] = -1
            tempy[tempy == self.unique_val[1]] = 1
            self.train(X, tempy)
        else:
            print("more or less than 2 clases")


    def predict(self,X):
        if(len(self.classes) == 2):
            Y = self.test(X)
            Y[Y == -1] = self.unique_val[0]
            Y[Y == 1] = self.unique_val[1]
            return Y
        else:
            print("more or less than 2 clases")

def read_dataset(Normalize = 1):
    train = pd.read_csv('Dataset/Titanic/train.csv')
    train_y = train['Survived']
    train = train.drop(['Survived','Name','Ticket','Cabin'] , axis=1)
    train['Sex'] = train['Sex'].replace(['male','female'] , (1,0))
    train['Embarked'] = train['Embarked'].replace(['C','S','Q'] , (0,1,2))
    train_x = np.asarray(train)
    train_y = np.asarray(train_y)
    train_x = np.nan_to_num(train_x)
    train_x, test_x , train_y,test_y = train_test_split(train_x, train_y,test_size=0.2, random_state=0)
    if Normalize == 1:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    return train_x ,train_y ,test_x ,test_y

train_x ,train_y ,test_x ,test_y = read_dataset()

Svm = svm(learning_rate=0.01, epochs=50)

Svm.fit(train_x ,train_y)

predict_y = Svm.predict(test_x)

correct = np.sum(predict_y == test_y)

print("%d out of %d predictions correct" % (correct, len(predict_y)))

print("accuracy = ", correct / len(predict_y) * 100)