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

    def sort_data(self,x,y):
        y = np.reshape(y,(len(y) , 1))
        data = np.concatenate((x , y),axis=1)
        data = sorted(data, key=lambda coul: coul[len(data[0])-1])
        data = np.asarray(data)
        x = data[:,:(len(x[0]))]
        y = data[:,len(x[0]):]
        y = np.reshape(y,(len(y)))
        return x , y

    def train_multi(self,X,Y):
        X , Y = self.sort_data(X,Y)
        u, self.classes = np.unique(Y, return_counts=True)
        self.weights = []
        self.biasies = []
        indi = 0
        for i in range(0 , len(self.classes)):
            arrW = []
            arrB = []
            indk = indi + self.classes[i]
            for k in range(i+1,len(self.classes)):
                temp_x = np.concatenate((X[indi:indi + self.classes[i]] , X[indk:indk + self.classes[k]]) , axis=0)
                temp_y = np.concatenate((Y[indi:indi + self.classes[i]] , Y[indk:indk + self.classes[k]]) , axis=0)
                temp_y[temp_y == i + 1] = 1
                temp_y[temp_y == k + 1] = -1
                self.train(temp_x, temp_y)
                arrW.append(self.weight)
                arrB.append(self.bias)
                indk += self.classes[k]
            self.weights.append(arrW)
            self.biasies.append(arrB)
            indi += self.classes[i]
        return self.weights , self.biasies

    def predict(self,X):
        return np.sign(X.dot(self.weight)+self.bias)

    def predict__(self,X,Weight , Bias):
        return np.sign(X.dot(Weight)+Bias)

    def predict_mult(self,X):
        predict_y = []
        for x in X:
            arr = np.zeros(len(self.classes))
            for i in range(0, len(self.classes)):
                for k in range(0, len(self.classes)-(i+1)):
                    Y = self.predict__(x,self.weights[i][k],self.biasies[i][k])
                    if  Y == 1:
                        arr[i] +=1
                    elif Y == -1:
                        arr[k+i+1] +=1
            predict_y.append(np.argmax(arr)+1)
        return predict_y

    def fit(self, X, Y):
        self.unique_val, self.classes = np.unique(Y, return_counts=True)
        if len(self.classes) > 2:
            self.train_multi(X, Y)
        else:
            tempy = np.copy(Y)
            tempy[ tempy == self.unique_val[0]] = -1
            tempy[ tempy == self.unique_val[1]] = 1
            self.train(X, tempy)

    def test(self,X):
        if(len(self.classes) > 2):
            return self.predict_mult(X)
        else:
            Y = self.predict(X)
            Y[Y ==  -1 ] = self.unique_val[0]
            Y[Y == 1 ] = self.unique_val[1]
            return Y

def read(Normalize = 1):
    train = pd.read_csv('../Dataset/Iris/Iris.csv')
    train['Species'] = train['Species'].replace(["Iris-setosa","Iris-versicolor","Iris-virginica"] , (1,2,3))
    train_y = train['Species']
    train = train.drop(['Species','Id'] , axis=1)
    train_x = np.asarray(train)
    train_y = np.asarray(train_y)
    train_x = np.nan_to_num(train_x)
    train_x, test_x , train_y,test_y = train_test_split(train_x, train_y,test_size=0.2, random_state=0)
    if Normalize == 1:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    return train_x ,train_y ,test_x ,test_y

train_x ,train_y ,test_x ,test_y = read()

sv = svm(learning_rate=0.001, epochs=2000)

sv.fit(train_x ,train_y)

predict_y = sv.test(test_x)

correct = np.sum(predict_y == test_y)

print("%d out of %d predictions correct" % (correct, len(predict_y)))

print("accuracy = ", correct / len(predict_y) * 100)
