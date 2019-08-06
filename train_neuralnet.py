# coding: utf-8
import numpy as np

from two_layer_net import TwoLayerNet

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import matplotlib.pyplot as plt

boston = load_boston()

(x_train, x_test, y_train, y_test) = train_test_split(boston.data, boston.target, test_size=0.3, random_state=0)

#xを正規化
sscaler_x = preprocessing.StandardScaler() 
sscaler_x.fit(x_train)
x_train= sscaler_x.transform(x_train)
x_test= sscaler_x.transform(x_test)

#yの検証用データを保存
y_test_original=y_test.reshape(-1, 1)

#yを正規化
sscaler_y = preprocessing.StandardScaler() 
sscaler_y.fit(y_train.reshape(-1, 1))
y_train= sscaler_y.transform(y_train.reshape(-1, 1))
y_test= sscaler_y.transform(y_test.reshape(-1, 1))

network = TwoLayerNet(input_size=13, hidden_size=50, output_size=1)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    
    # 勾配
    grad = network.gradient(x_batch, y_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, y_batch)
    train_loss_list.append(loss)
    
y_predict=network.predict(x_test)
y_predict=sscaler_y.inverse_transform(y_predict)
plt.plot(train_loss_list)
print('finish')