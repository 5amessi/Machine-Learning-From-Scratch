import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
def read(Normalize = 1):
    train_x = np.asarray(pd.read_csv('Dataset/TrainX_pca.csv'))
    train_y = np.asarray(pd.read_csv('Dataset/TrainY_pca.csv'))
    train_y = np.reshape(train_y , (len(train_y)))
    test_x  = np.asarray(pd.read_csv('Dataset/TestX_pca.csv'))
    test_y  = np.asarray(pd.read_csv('Dataset/TestY_pca.csv'))
    test_y  = np.reshape(test_y , (len(test_y)))
    if Normalize == 1:
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)
    return train_x ,train_y ,test_x ,test_y