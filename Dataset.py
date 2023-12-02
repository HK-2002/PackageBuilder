import numpy as np
import pandas as pd
import paddle
from sklearn.preprocessing import MinMaxScaler

class MyDataset(paddle.io.Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        self.data = paddle.to_tensor(x, dtype='float32')
        self.label = paddle.to_tensor(y, dtype='float32')

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)

class my_dataset():
    def __init__(self,x_data,y_data,ratio):
        self.ratio=ratio

        data_num=int(x_data.shape[0]*self.ratio)
        self.x_train=x_data[:data_num]
        self.x_test=x_data[data_num:]
        self.y_train=y_data[:data_num]
        self.y_test=y_data[data_num:]

        self.train_dataset = MyDataset(self.x_train,self.y_train)
        self.test_dataset = MyDataset(self.x_test,self.y_test)