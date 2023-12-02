import pandas as pd
import numpy as np
import paddle
from Model import Att
from Dataset import my_dataset
from TestBenchh_of_ML import ReadingFile,PreProcessingX,PreProcessingY
from scipy import stats
from matplotlib import pyplot as plt

class MyLoss(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, label):
        loss=0
        num=x.shape[0]
        for i in range(num):
            loss+=1-stats.spearmanr(x[i],label[i])[0]
        loss=paddle.to_tensor(loss/num)
        return loss

X = PreProcessingX(index_nums=6,data_set=ReadingFile()) #输入样本数据
Y = PreProcessingY() #输出样本数据

ratio=0.8
dt=my_dataset(X,Y,ratio)
tmp=Att()
paddle.summary(tmp,dt.x_train.shape)

model=paddle.Model(Att())
model.prepare(paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters()),
              MyLoss())
model.fit(dt.train_dataset, epochs=100, batch_size=8, verbose=1)
result=model.predict(dt.test_dataset)

ans=[]
index=[]
for i in range(len(dt.test_dataset)):
    y_pre=result[0][i][0]
    y_real=dt.test_dataset[i][1]
    ans.append(stats.spearmanr(y_pre,y_real)[0])
    index.append(i)
plt.scatter(index,ans)
plt.show() #测试集中每条数据真实值和预测值的相关系数