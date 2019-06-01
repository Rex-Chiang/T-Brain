# 載入需要的套件
import os
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf

'''數據導入'''
# 設定 data_path
dir_data = '/home/rex/桌面/T-Brain/data/'
Train = os.path.join(dir_data, 'train.csv')
Test = os.path.join(dir_data, 'test.csv')

# 讀取檔案
Train_data = pd.read_csv(Train)
Test_data = pd.read_csv(Test)

train_Y = np.log1p(Train_data['total_price'])
ids = Test_data['building_id']

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
test_X = df[train_num:]

with tf.Session() as sess:
   
    saver = tf.train.import_meta_graph("/home/rex/桌面/T-Brain/DNN/2019-06-01/2/test.meta")
   
    saver.restore(sess, "/home/rex/桌面/T-Brain/DNN/2019-06-01/2/test")
    graph = tf.get_default_graph()
    predict = tf.get_collection('predict')[0]
    X = graph.get_operation_by_name("X").outputs[0]

    print("模型恢復成功！")

    pred = sess.run(predict, feed_dict = {X : test_X})
    
    pred_price = np.expm1(pred)
    
    submit = pd.DataFrame(ids)
    submit['total_price'] = pred_price.astype(np.int32)
    submit.to_csv("2019-06-01.csv",index=False)