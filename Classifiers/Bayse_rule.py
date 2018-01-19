import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class bayes_rule():

    def fit(self,train_x,train_y):
        train_x , train_y = self.sort_data(train_x,train_y)
        self.classes, self.prior = np.unique(train_y, return_counts=True)
        self.prior_probability = self.prior / len(train_y)
        self.train_mean = np.zeros((len(self.classes),np.shape(train_x)[1]))
        self.train_variance = np.zeros((len(self.classes),np.shape(train_x)[1]))
        start = 0
        for i in range(len(self.classes)):
            temp_trainx = train_x[start:start + self.prior[i]]
            self.train_mean[i] = np.mean(temp_trainx, axis=0)
            self.train_variance[i] = np.sqrt(np.sum(((temp_trainx - self.train_mean[i]) ** 2), axis=0) / (max(len(temp_trainx) - 1, 1)))
            start += self.prior[i]

    def sort_data(self,x,y):
        y = np.reshape(y,(len(y) , 1))
        data = np.concatenate((x , y),axis=1)
        data = sorted(data, key=lambda coul: coul[len(data[0])-1])
        data = np.asarray(data)
        x = data[:,:(len(x[0]))]
        y = data[:,len(x[0]):]
        y = np.reshape(y,(len(y)))
        return x , y

    def predict(self,test_x):
        likelihood = np.zeros(len(self.classes))
        predicted_y = []
        for i in range(len(test_x)):
            for c in range(len(self.classes)):
                if (np.count_nonzero(self.train_variance[c]) == 0):
                    likelihood[c] = 0
                else:
                    x = (1 / (np.sqrt(2 * (22 / 7)) * self.train_variance[c]))
                    p1 = (test_x[i] - self.train_mean[c]) ** 2
                    p2 = (self.train_variance[c]) ** 2
                    p3 = (-0.5 * p1) / p2
                    x2 = np.exp(p3)  # 5amessi
                    likelihood[c] = self.mult(x * x2)
            Px = np.sum(likelihood * self.prior_probability)
            pwi_given_x = [(likelihood[i] * self.prior_probability[i] / Px , int(k)) for i,k in enumerate(self.classes)]
            pwi_given_x = sorted(pwi_given_x, reverse=True)
            predicted_y.append(pwi_given_x[0][1])
        return predicted_y

    def mult(self,arr):
        a=1
        for i in arr:
            a *= i
        return (a)

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

bl = bayes_rule()

bl.fit(train_x,train_y)

predict_y = bl.predict(test_x)

correct = np.sum(predict_y == test_y)

print("%d out of %d predictions correct" % (correct, len(predict_y)))

print("accuracy = ", correct / len(predict_y) * 100)


