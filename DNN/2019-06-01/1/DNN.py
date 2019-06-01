#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:07:41 2019

@author: rex
"""

# 載入需要的套件
import os
import numpy as np 
import pandas as pd
import copy
import time
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf

'''數據導入'''
# 設定 data_path
dir_data = '/tmp/tbrain/DNN/data/'
Train = os.path.join(dir_data, 'train.csv')
Test = os.path.join(dir_data, 'test.csv')

# 讀取檔案
Train_data = pd.read_csv(Train)
Test_data = pd.read_csv(Test)

train_Y = np.log1p(Train_data['total_price'])
ids = Test_data['building_id']

tp = copy.deepcopy(np.log1p(Train_data['total_price']))

Train_data = Train_data.drop(['building_id', 'total_price'] , axis=1)
Test_data = Test_data.drop(['building_id'] , axis=1)

df = pd.concat([Train_data,Test_data])

'''數據前處理'''
df = df.fillna(-1)
train_num = train_Y.shape[0]
train_x = df[:train_num].values

estimator = RandomForestRegressor()
estimator.fit(train_x, train_Y)
feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)
high_feature = list(feats[:37].index)

df = df[high_feature].values

MMEncoder = MaxAbsScaler()
df = MMEncoder.fit_transform(df)
train_X = df[:train_num]

X_train_data = train_X[:int(train_num*0.9)]
X_test_data = train_X[int(train_num*0.9):]
Y_train_data = train_Y[:int(train_num*0.9)].values.reshape(-1, 1)
Y_test_data = train_Y[int(train_num*0.9):].values.reshape(-1, 1)

n_input = X_train_data.shape[1]
n_class = Y_train_data.shape[1]


learning_rate= 0.0001
training_epochs = 1000 # 訓練迭代數
batch_size = 32 # 批數量
Now_time= time.time() # 訓練開始時間

# 神經網路權重初始化
def weight_variable(shape, name):
   
    return tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer())

# 神經網路偏置初始化
def bias_variable(shape, name):
    
    return tf.get_variable(name, shape, initializer = tf.zeros_initializer())

# placeholder用於定義過程，在之後執行的時再賦與具體的值
X = tf.placeholder(tf.float32, [None, n_input], name="X")
Y = tf.placeholder(tf.float32, [None, n_class], name="Y")

'''神經網路建立'''
w1= tf.get_variable('w1', [n_input, 120], initializer = tf.contrib.layers.xavier_initializer())
w2= tf.get_variable('w2', [120, 60], initializer = tf.contrib.layers.xavier_initializer())
w3= tf.get_variable('w3', [60, n_class], initializer = tf.contrib.layers.xavier_initializer())

b1= tf.get_variable('b1', [120], initializer = tf.zeros_initializer())
b2= tf.get_variable('b2', [60], initializer = tf.zeros_initializer())
b3= tf.get_variable('b3', [n_class], initializer = tf.zeros_initializer())

def multiplayer_perceptron(x):
    layer1 = tf.add(tf.matmul(x, w1), b1)    
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(layer1, w2), b2)   
    layer2 = tf.nn.relu(layer2)
    out = tf.add(tf.matmul(layer2, w3), b3)

    return out

# 最後預測结果
prediction =  multiplayer_perceptron(X)
tf.add_to_collection('predict', prediction)

# Cost Function
with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.squared_difference(prediction, Y))
# Optimization 
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize all variables
init = tf.global_variables_initializer()

cost_history= []
test_history= []

'''開始訓練'''
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    
    for epoch in range(0, training_epochs+1):
        avg_cost = 0
        total_batch = int(X_train_data.shape[0] / batch_size)
    
        for i in range(total_batch):
            _, cost = sess.run([optimizer, loss], feed_dict={X: X_train_data[i*batch_size : (i+1)*batch_size, :], 
                                                          Y: Y_train_data[i*batch_size : (i+1)*batch_size, :]})
            # 計算訓練誤差
            avg_cost += cost / total_batch
        
        # 計算測試誤差
        TestLoss = loss.eval({X: X_test_data, Y: Y_test_data})
        cost_history.append(avg_cost)
        test_history.append(TestLoss)
        # 每迭代500次輸出一次
        if epoch % 100 == 0:
            print('Epoch:', '%04d' % (epoch), 'loss=', '{:.9f}'.format(avg_cost))
            print('Epoch:', '%04d' % (epoch), "TestLoss=", '{:.9f}'.format(TestLoss))

    print('Opitimization Finished!\n')
    
    End_time= time.time() # 訓練結束時間
    # 輸出總訓練時間
    print("Time: ", '{:.3f}'.format((End_time-Now_time)/60), "minutes")


 # 繪出交叉比對結果
    line1, = plt.plot(np.arange(len(cost_history)), cost_history, "b", label= "Training loss")
    line2, = plt.plot(np.arange(len(test_history)), test_history, "r",label= 'Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(color='g',linestyle='--', linewidth=1,alpha=0.4)
    plt.tight_layout()
    plt.show()
    
    # 保存模型
    saver.save(sess, "/tmp/tbrain/test")
    print("保存模型成功！")
