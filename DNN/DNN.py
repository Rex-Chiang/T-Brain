import os
import numpy as np 
import pandas as pd
import copy
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from scipy import stats
import matplotlib.pyplot as plt

learning_rate= 0.001
training_epochs = 1000 # 訓練迭代數
batch_size = 32 # 批數量
Now_time= time.time() # 訓練開始時間
'''數據導入'''
# 設定 data_path
dir_data = '/tmp/tbrain/DNN/data/'
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

#df['txn_dt'] = np.log1p(df['txn_dt'])
#df['building_complete_dt'] = np.log1p(df['building_complete_dt'])
#df['village'] = np.log1p(df['village'])
#df['II_5000'] = np.log1p(df['II_5000'])
#df['II_10000'] = np.log1p(df['II_10000'])
#df['town_population'] = np.log1p(df['town_population'])
#df['town_population_density'] = np.log1p(df['town_population_density'])
#df['XIV_5000'] = np.log1p(df['XIV_5000'])
#df['XIV_10000'] = np.log1p(df['XIV_10000'])

df = df.fillna(df.mean())

print(type(df))
columns = list(df.columns)
MM = MinMaxScaler()

for col in columns:
    if df[col].mean() >=100:
        df[col] = MM.fit_transform(df[col].values.reshape(-1, 1))

#df = MM.fit_transform(df.values.reshape(-1, 1))

train_num = train_Y.shape[0]
train_X = df[:train_num].values

# 隨機劃分為訓練集及測試集
X_train_data = train_X[:int(train_num*0.9)]
X_test_data = train_X[int(train_num*0.9):]
Y_train_data = train_Y[:int(train_num*0.9)]
Y_test_data = train_Y[int(train_num*0.9):]

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

# placeholder用於定義過程，在之後執行的時再賦與具體的值
# 輸入數據维度：16
X = tf.placeholder(tf.float32, [None, n_input], name="X")

# 輸出數據維度：5
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
    
    #fc_mean, fc_var = tf.nn.moments(layer1,axes=[0],)
    #scale = tf.Variable(tf.ones([120]))
    #shift = tf.Variable(tf.zeros([120]))
    #epsilon = 0.001
    #layer1 = tf.nn.batch_normalization(layer1, fc_mean, fc_var, shift, scale, epsilon)
    
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(layer1, w2), b2)
    
    #fc_mean, fc_var = tf.nn.moments(layer2,axes=[0],)
    #scale = tf.Variable(tf.ones([50]))
    #shift = tf.Variable(tf.zeros([50]))
    #epsilon = 0.001
    #layer2 = tf.nn.batch_normalization(layer2, fc_mean, fc_var, shift, scale, epsilon)
    
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
    saver.save(sess, "/tmp/tbrain/DNN/test")
    print("保存模型成功！")
