import os
import numpy as np 
import pandas as pd
import copy
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt


'''數據導入'''
# 設定 data_path
dir_data = '/home/rex/桌面/T-Brain/data/'
Train = os.path.join(dir_data, 'train.csv')
Test = os.path.join(dir_data, 'test.csv')
# 讀取檔案
Train_data = pd.read_csv(Train)
Test_data = pd.read_csv(Test)

Corr_train = copy.deepcopy(Train_data)
corr_tar = Corr_train.corr()['total_price']
NanCol = corr_tar.sort_values(na_position='first')[:8]

train_Y = np.log1p(Train_data['total_price'])
train_Y = train_Y.values.reshape(-1,1)
ids = Test_data['building_id']
Train_data = Train_data.drop(['building_id', 'total_price'] , axis=1)
Test_data = Test_data.drop(['building_id'] , axis=1)

for col in NanCol.index:
    Train_data = Train_data.drop(col, axis=1)
    Test_data = Test_data.drop(col , axis=1)

df = pd.concat([Train_data,Test_data])

# 再把只有 2 值 (通常是 0,1) 的欄位去掉
#new_columns = list(df.columns[list(df.apply(lambda x:len(x.unique())!=2 ))])
#df = df[new_columns]

df = df.fillna(df.median()).values

train_num = train_Y.shape[0]
test_X = df[train_num:]

with tf.Session() as sess:
   
<<<<<<< HEAD
    saver = tf.train.import_meta_graph("/home/rex/桌面/T-Brain/CNN/05-09/test.meta")
   
    saver.restore(sess, "/home/rex/桌面/T-Brain/CNN/05-09/test")
=======
    saver = tf.train.import_meta_graph("/home/rex/桌面/T-Brain/CNN/test.meta")
   
    saver.restore(sess, "/home/rex/桌面/T-Brain/CNN/test")
>>>>>>> 83f28894daa65319b76715c2effc73b0bb158e50
    graph = tf.get_default_graph()
    predict = tf.get_collection('predict')[0]
    X = graph.get_operation_by_name("X").outputs[0]

    print("模型恢復成功！")

    pred = sess.run(predict, feed_dict = {X : test_X})
    
    pred_price = np.expm1(pred)
<<<<<<< HEAD
=======
    for i in range(len(pred_price)):
        if pred_price[i]<0:
            print(pred_price[i])
            pred_price[i] = abs(pred_price[i])
>>>>>>> 83f28894daa65319b76715c2effc73b0bb158e50
    
    submit = pd.DataFrame(ids)
    submit['total_price'] = pred_price.astype(np.int32)
    submit.to_csv("2019-05-09.csv",index=False)