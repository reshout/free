import tensorflow as tf
import numpy as np
import pandas
from pandas import DataFrame
from os import listdir


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
        X[idx - s_idx] = M[idx - p_days + 1:idx + 1].flatten() / 10000
        b_close = M[idx][0]
        f_close = M[idx + f_days][0]
        y[idx - s_idx] = (f_close / b_close - 1) if f_close >= b_close else -(1 - f_close / b_close)
    return (X, y)


kospi_items = get_items('KOSPI')
(X, y) = gen_data('KOSPI', kospi_items[0][0], kospi_items[0][1])

X_data = np.float32(X[0:100]).transpose()
y_data = np.float32(y[0:100])

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

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)

for step in range(0, 200):
    session.run(train)
    print(step, session.run(loss))


