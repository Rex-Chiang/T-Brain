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
dir_data = '/home/rex/桌面/T-Brain/DNN/data/'
Train = os.path.join(dir_data, 'train.csv')
Test = os.path.join(dir_data, 'test.csv')
# 讀取檔案
Train_data = pd.read_csv(Train)
Test_data = pd.read_csv(Test)

train_Y = np.log1p(Train_data['total_price'])
train_Y = train_Y.values.reshape(-1,1)
ids = Test_data['building_id']
Train_data = Train_data.drop(['building_id', 'total_price'] , axis=1)
Test_data = Test_data.drop(['building_id'] , axis=1)

df = pd.concat([Train_data,Test_data])

df['building_area'] = df['building_area'].clip(0.3, 20)

df['parking_price'].replace({np.nan: df['parking_price'].mean()}, inplace = True)
df['parking_price'] = np.log1p(df['parking_price'])

df['parking_area'].replace({np.nan: df['parking_area'].mean()}, inplace = True)
df['parking_area'] = df['parking_area'].clip(0.35, 30)

df['land_area'] = df['land_area'].clip(2.22, 100)
df['land_area'] = stats.boxcox(df['land_area'], lmbda=0.5)

df = df.fillna(df.mean())

columns = list(df.columns)
MM = MinMaxScaler()

for col in columns:
    if df[col].mean() >=100:
        df[col] = MM.fit_transform(df[col].values.reshape(-1, 1))

train_num = train_Y.shape[0]
test_X = df[train_num:]

with tf.Session() as sess:
   
    saver = tf.train.import_meta_graph("/home/rex/桌面/T-Brain/DNN/2019-05-12-1-USE/test.meta")
   
    saver.restore(sess, "/home/rex/桌面/T-Brain/DNN/2019-05-12-1-USE/test")
    graph = tf.get_default_graph()
    predict = tf.get_collection('predict')[0]
    X = graph.get_operation_by_name("X").outputs[0]

    print("模型恢復成功！")

    pred = sess.run(predict, feed_dict = {X : test_X})
    
    pred_price = np.expm1(pred)
    
    submit = pd.DataFrame(ids)
    submit['total_price'] = pred_price.astype(np.int32)
    submit.to_csv("2019-05-13.csv",index=False)