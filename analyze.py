import tensorflow as tf
import numpy as np
import pandas
from pandas import DataFrame
from os import listdir
from random import shuffle


def get_items(market):
    items = []
    for file in listdir('./{}'.format(market)):
        if file.endswith('.csv'):
            (code, name) = file.split(sep='.')[0].split(sep='-', maxsplit=1)
            items.append((code, name))
    return items


def gen_data(market, code, name):
    stock_data = pandas.read_csv('./{}/{}-{}.csv'.format(market, code, name))
    stock_data = DataFrame(stock_data, columns=['Close'])
    M = stock_data.as_matrix()
    length = M.shape[0] 
    p_days = 30
    f_days = 5
    s_idx = p_days - 1
    e_idx = length - 1 - f_days
    X = np.zeros((e_idx - s_idx + 1, p_days * 1))
    y = np.zeros(e_idx - s_idx + 1)
    for idx in range(s_idx, e_idx + 1):
        b_close = M[idx][0]
        f_close = M[idx + f_days][0]
        X[idx - s_idx] = M[idx - p_days + 1:idx + 1].flatten() / b_close
        y[idx - s_idx] = (f_close / b_close - 1) if f_close >= b_close else -(1 - f_close / b_close)
    return (X, y)


kospi_items = get_items('KOSPI')
shuffle(kospi_items)
(X, y) = gen_data('KOSPI', kospi_items[0][0], kospi_items[0][1])

X_data = np.float32(X).transpose()
y_data = np.float32(y)

print(X_data.shape)
print(X_data)
print(y_data.shape)
print(y_data)

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, X_data.shape[0]]))
y_predict = tf.matmul(W, X_data) + b
loss = tf.reduce_mean(tf.square(y_predict - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

(X2, y2) = gen_data('KOSPI', kospi_items[1][0], kospi_items[1][1])
X2_data = np.float32(X2).transpose()
y2_data = np.float32(y2)
y2_predict = tf.matmul(W, X2_data) + b
loss2 = tf.reduce_mean(tf.square(y2_predict - y2_data))

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

for step in range(0, 100):
    session.run(train)

print(session.run(loss))
print(session.run(loss2))

print(y2_data)
print(session.run(y2_predict))