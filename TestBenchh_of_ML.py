# -*- coding: utf-8 -*-
# from keras.src.layers.serialization import activation
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

# from keras.layers import Dense, Input, Dropout, LSTM, Flatten
# from keras.optimizers import SGD,Adam
# from keras.models import Model, load_model
# from keras.callbacks import ModelCheckpoint
# from keras import Sequential

#tf.compat.v1.disable_eager_execution()
#import tensorflow_probability as tfp

from sklearn.metrics import r2_score

def ReadingFile(dir_folder:str = "./TestBench1/",
                index_nums:int = 6,
                index_name_list:list = ["P-E","Price","Vol","STD","AMP","Return"],
                year:str = "2022"):
    index_dir_list = [] #存储因子目标地址的
    for i in range(index_nums):
        index_dir_list.append(dir_folder+"IM_"+index_name_list[i]+"_"+year+".csv")
    data_set = []
    for i in range(index_nums):
        data_set.append(pd.read_csv(index_dir_list[i]).bfill().ffill())
    return data_set

def PreProcessingX(index_nums:int = 6,
                   data_set:list[pd.DataFrame] = [...],
                   time_series_namer:str = "trade_date"):
    # multi_check_shape(CheckReturnData=False,
    #                   index_data_list=data_set)
    for i in range(index_nums):
        data_set[i] = data_set[i].set_index(keys=time_series_namer).T
    time_series = data_set[i].columns
    X_array = []
    for time in time_series:
        #print(time)
        temp_array = []
        for data in data_set:
            data[time] = data[time]/(data[time].max()-data[time].min())
            arr = data[time].array
            arr_mean_fill = np.where(np.isnan(arr), np.nanmean(arr), arr)
            temp_array.append(arr_mean_fill)
        X_array.append(temp_array)
    return np.array(X_array[:-1])

def PreProcessingY(data:pd.DataFrame = pd.read_csv("./TestBench1/IM_RANK_2022.csv"),
                   time_series_namer:str = "trade_date"):
    data = data.set_index(keys=time_series_namer).T.bfill().ffill()
    time_series = data.columns
    Y_array = []
    for time in time_series:
        data[time] = data[time]/(data[time].max()-data[time].min())
        Y_array.append(data[time].array)
    return np.array(Y_array[1:])

# @tf.autograph.experimental.do_not_convert
# def custom_loss(y_true, y_pred):
#     # Reshape y_true and y_pred to 1D arrays
#     print(y_true.shape,y_pred.shape)
#     #with tf.compat.v1.Session() as sess: print(y_true.eval())
#     y_true = tf.reshape(y_true, [-1])
#     y_pred = tf.reshape(y_pred, [-1])
#     #print(y_true.shape,y_pred.shape)
#     # Calculate Spearman rank correlation coefficient
#     correlation, _ = tf.py_function(spearmanr, [y_true, y_pred], [tf.float64, tf.float64])

#     # Negative of correlation as we want to minimize the negative correlation
#     loss = -correlation

#     return loss
    

# X = PreProcessingX(index_nums=6,
#                    data_set=ReadingFile()) #输入样本数据

# Y = PreProcessingY() #输出样本数据

# print(X.shape,Y.shape)

# print((True in np.isnan(X)))
# print((True in np.isnan(Y)))
# """
# input_layer = Flatten(input_shape=(6,900),dtype='float32')
# dense1 = Dense(units=512,activation='relu')(input_layer)
# dense2 = Dense(units=256,activation='relu')(dense1)
# dense3 = Dense(units=128,activation='relu')(dense2)

# dropout_layer = Dropout(0.2)(dense3)
# output_layer = Dense(units=900,activation='linear')(dropout_layer)
# """
# ts_model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(6, 900)),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(900,activation='linear')
# ])

# #ts_model = Model(inputs=input_layer,outputs=output_layer)

# ts_model.compile(loss="mean_squared_error",optimizer=Adam(learning_rate=0.0001))
# #ts_model.compile(loss=custom_loss,optimizer=Adam(learning_rate=0.0001))
# ts_model.summary()

# save_weights_at = os.path.join('MLP','PRSA_data_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
# save_best = ModelCheckpoint(save_weights_at,monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='min',save_freq='epoch')
# ts_model.fit(x=X,y=Y,batch_size=16,epochs=300,verbose=1,callbacks=[save_best],validation_split=0.2,shuffle=True)