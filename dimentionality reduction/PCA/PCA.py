import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from softmax_regression_script import SoftmaxRegression

class PCA():
    def __init__(self,Normalize = 0,cut_in = 0.90):
        self.Normalize = Normalize
        self.cut_in = cut_in

    def Train(self,data):
        if self.Normalize == 1:
            self.scaler = MinMaxScaler()
            data = self.scaler.fit_transform(data)
        self.mean = np.mean(data , axis=0)
        mean_adj_dataset = data - self.mean
        cov = (mean_adj_dataset).T.dot(mean_adj_dataset) / (data.shape[0]-1)
        cov = np.asarray(cov,float)
        eigen_value , eigen_vector = np.linalg.eig(cov)
        eigen_value = np.asarray(eigen_value,float)
        eigen_vector = np.asarray(eigen_vector,float)
        sum = np.sum(eigen_value)
        self.var_exp = [(i / sum) for i in sorted(eigen_value, reverse=True)]
        self.var_exp = np.asarray(np.cumsum(self.var_exp))
        Pair_eigne = [(abs(eigen_value[i]),eigen_vector.T[i]) for i in range(len(eigen_value))]
        Pair_eigne = sorted(Pair_eigne, key=lambda eigne: eigne[0],reverse=True)
        ind = np.where(self.var_exp >= self.cut_in)[0][0]
        self.New_Eigen_Vectors = [Pair_eigne[i][1] for i in range(ind)]
        self.New_Eigen_Vectors = np.transpose(self.New_Eigen_Vectors)
        final_data = mean_adj_dataset.dot(self.New_Eigen_Vectors)
        return final_data

    def Test(self , data):
        if self.Normalize == 1:
            data = self.scaler.transform(data)
        return (data - self.mean).dot(self.New_Eigen_Vectors)

    def plot(self):
        plt.plot(range(1, len(self.var_exp) + 1), self.var_exp)
        plt.xlabel('Num of EigVec')
        plt.ylabel('Var')
        plt.title('5amessi')
        plt.tight_layout()
        plt.show()

data = [[1,1,4,5] , [1,2,4,4]]
data = np.asarray(data)
mean = np.mean(data , axis=1)
print(mean)
covv = np.cov(data)
print(covv)
eigen_value , eigen_vector = np.linalg.eig(covv)
eigen_value = np.asarray(eigen_value,float)
eigen_vector = np.asarray(eigen_vector,float)
print(eigen_value)
print(eigen_vector)
print(data-np.transpose(mean))

"""
width , high = 50 , 50
#read_images
train = [(i,np.reshape(np.asarray(Image.open(img).convert('L').resize((width,high))),(width*high))) for i in range(1,29) for img in glob.glob("../../Dataset/Faces/data/"+str(i)+"/*.jpg")]

train_x = [train[i][1] for i in range(len(train))]

train_y = [train[i][0] for i in range(len(train))]

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=50)

pca = PCA(Normalize=1,cut_in=0.90)

train_x = pca.Train(np.asarray(train_x))

test_x = pca.Test(np.asarray(test_x))

print(np.shape(train_x))

print(np.shape(test_x))

softmax = SoftmaxRegression(learning_rate=0.01, epochs=1000)

softmax.fit(train_x,train_y)

softmax.plot()

predict_y = softmax.predict(test_x)

correct = np.sum(predict_y == test_y)

print("%d out of %d predictions correct" % (correct, len(predict_y)))

print("accuracy = ", correct / len(predict_y) * 100)
"""