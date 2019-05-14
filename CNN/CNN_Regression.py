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

learning_rate= 0.01
training_epochs = 1000 # 訓練迭代數
batch_size = 32 # 批數量
Now_time= time.time() # 訓練開始時間
'''數據導入'''
# 設定 data_path
dir_data = './data/'
Train = os.path.join(dir_data, 'train.csv')
Test = os.path.join(dir_data, 'test.csv')
# 讀取檔案
Train_data = pd.read_csv(Train)
Test_data = pd.read_csv(Test)

Corr_train = copy.deepcopy(Train_data)
corr_tar = Corr_train.corr()['total_price']
NanCol = corr_tar.sort_values(na_position='first')[:8]

train_Y = np.log1p(Train_data['total_price'])
train_Y = train_Y.reshape(-1,1)[:100]
ids = Test_data['building_id']
Train_data = Train_data.drop(['building_id', 'total_price'] , axis=1)
Test_data = Test_data.drop(['building_id'] , axis=1)

for col in NanCol.index:
    Train_data = Train_data.drop(col, axis=1)
    Test_data = Test_data.drop(col , axis=1)

df = pd.concat([Train_data,Test_data])

<<<<<<< HEAD
=======
# 再把只有 2 值 (通常是 0,1) 的欄位去掉
#new_columns = list(df.columns[list(df.apply(lambda x:len(x.unique())!=2 ))])
#df = df[new_columns]

>>>>>>> 83f28894daa65319b76715c2effc73b0bb158e50
df = df.fillna(df.median()).values

train_num = train_Y.shape[0]
train_X = df[:train_num]

# 隨機劃分為訓練集及測試集
X_train_data, X_test_data, Y_train_data, Y_test_data= train_test_split(train_X, train_Y,train_size=0.9, random_state=33)

n_input = X_train_data.shape[1]
n_class = Y_train_data.shape[1]

'''數據前處理'''
# 對輸入資料的訓練集及測試集作數據標準化
#standard_x= preprocessing.StandardScaler()
#train_x_data= standard_x.fit_transform(train_x_data)
#test_x_data= standard_x.transform(test_x_data)

# 對輸出資料的訓練集及測試集作數據標準化
#standard_y= preprocessing.StandardScaler()
#train_y_data= standard_y.fit_transform(train_y_data.reshape(-1, 1))
#test_y_data= standard_y.transform(test_y_data.reshape(-1, 1))

'''神經網路框架設置'''
# 設定隨機之神經網路權重(常態分布)，只保留兩個標準差之内的點，
# 在兩個標準差之外的點就被捨去，然後重新抽樣，直到達到所設置的個數
# 神經網路權重初始化
def weight_variable(shape, name):
   
    return tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer())

# 神經網路偏置初始化
def bias_variable(shape, name):
    
    return tf.get_variable(name, shape, initializer = tf.zeros_initializer())

# 神經網路之卷積層的設置
def conv2d(x, W):
    # stride = [1, x_movement, y_movement, 1]，其中x_movement、y_movement為步長
    # padding='SAME'表示卷積之後長寬不變
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# placeholder用於定義過程，在之後執行的時再賦與具體的值
# 輸入數據维度：16
X = tf.placeholder(tf.float32, [None, n_input], name="X")

# 輸出數據維度：5
Y = tf.placeholder(tf.float32, [None, n_class], name="Y")

# dropout的比例
#keep_prob = tf.placeholder(tf.float32)

# 原始一維數據16變成二维圖片4*4
x_image= tf.reshape(X, [-1, 15, 15, 1])

'''神經網路建立'''
# <conv1 layer>，第1層卷積層
# patch: 2x2, input size: 1, output size: 32, 每個像素變成32個像素
# patch為CNN濾波器的尺寸，也是圖片區塊的大小
W_conv1 = weight_variable([2, 2, 1, 16], "w1")
b_conv1 = bias_variable([16], "b1")
layer1= conv2d(x_image, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(layer1)

# <conv2 layer>，第2層卷積層
# patch: 2*2, input size: 14*14*32, output size: 14*14*64
W_conv2 = weight_variable([2, 2, 16, 32], "w2")
b_conv2 = bias_variable([32], "b2")
layer2= conv2d(h_conv1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(layer2)

# <fc1 layer>，full connection 全連接層1
# size: 4x4x64，高度為64的三維圖片，展開成512長的一維數組，降維處理
W_fc1 = weight_variable([(15*15)*32, 32], "w3")
b_fc1 = bias_variable([32], "b3")
h_flat = tf.reshape(h_conv2, [-1, (15*15)*32])
layer3= tf.matmul(h_flat, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(layer3)

# <fc2 layer>，full connection 全連接層2
# 512長的一維數組壓縮為長度為1的數組
W_fc2 = weight_variable([32, n_class], "w4")
b_fc2 = bias_variable([n_class], "b4")
# 最後預測结果
prediction =  tf.matmul(h_fc1, W_fc2) + b_fc2
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
    
    count=0
    
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
    saver.save(sess, "/home/rex/桌面/T-Brain/test")
    print("保存模型成功！")
