import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

df_1 = pd.read_csv('季度+年度数据-方案一.csv')
df_2 = pd.read_csv('年度数据-方案一.csv')

df_1.head()


#取出(年度+季度)数据的X和y
X1 = df_1.iloc[:,2:-1]
y1 = df_1.iloc[:,-1:]

#取出年度数据的X和y
X2 = df_2.iloc[:,2:-1]
y2 = df_2.iloc[:,-1:]

#取出季度数据的X和y
X3 = df_1.iloc[:,15:-1]
y3 = df_1.iloc[:,-1:]


import tensorflow as tf
# import torch.nn as nn
# import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# 定义全连接神经网络
class FCNN():
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.w1 = tf.Variable(np.random.normal(size=[input_dim, 256])/np.sqrt(256+0.0), dtype='float32')
        self.b1 = tf.Variable(np.zeros([1,256]), dtype='float32')
        self.w2 = tf.Variable(np.random.normal(size=[256, 128])/np.sqrt(128.0), dtype='float32')
        self.b2 = tf.Variable(np.zeros([1, 128]), dtype='float32')
        self.w3 = tf.Variable(np.random.normal(size=[128, 64])/np.sqrt(64.0), dtype='float32')
        self.b3 = tf.Variable(np.zeros([1, 64]), dtype='float32')
        self.w4 = tf.Variable(np.random.normal(size=[64, 32])/np.sqrt(32.0), dtype='float32')
        self.b4 = tf.Variable(np.zeros([1, 32]), dtype='float32')
        self.w5 = tf.Variable(np.random.normal(size=[32, 1]), dtype='float32')
        self.b5 = tf.Variable(np.zeros([1, 1]), dtype='float32')

        
    
    def forward(self, x):
        self.x1 = x@self.w1+self.b1
        self.x1 = tf.nn.relu(self.x1)
        self.x2 = self.x1@self.w2+self.b2
        self.x2 = tf.nn.relu(self.x2)
        self.x3 = self.x2@self.w3+self.b3
        self.x3 = tf.nn.relu(self.x3)
        self.x4 = self.x3@self.w4+self.b4
        self.x4 = tf.nn.relu(self.x4)
        y = self.x4@self.w5+self.b5
        return y
    # def backward(self, plosspy, batch_size):
    #     plosspw5 = tf.matmul(self.x4, plosspy, transpose_a=True)/batch_size
    #     plosspb5 = tf.reduce_mean(plosspy, axis=[0])
    #     plosspx4 = tf.matmul(plosspy, )
        
        
        
        
    



# 训练和评估模型的函数
def train_and_evaluate(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    fcnn = FCNN(input_dim)
    batch_size=32
    X = tf.compat.v1.data.Dataset.from_tensor_slices(X_train).repeat(100).batch(batch_size).make_one_shot_iterator().get_next()
    y = tf.compat.v1.data.Dataset.from_tensor_slices(y_train).repeat(100).batch(batch_size).make_one_shot_iterator().get_next()
    X = tf.cast(X, 'float32')
    y = tf.cast(y, 'float32')
    
    # X = tf.compat.v1.placeholder(tf.float32, shape=(None, input_dim))
    # y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
    y_pred = fcnn.forward(X)
    # print("y_pred.shape=", y_pred.shape)
    
    # 定义损失函数和优化器
    loss = tf.reduce_mean(tf.square(y_pred - y))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    # train_op = optimizer.minimize(loss)
    grads_and_vars = optimizer.compute_gradients(loss)
    print("grads_and_vars=", grads_and_vars)
    train_op = optimizer.apply_gradients(grads_and_vars)
    
    
    
    # X_ = tf.compat.v1.placeholder(tf.float32, shape=(None, input_dim))
    # y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
    # y_pred = fcnn.forward(X_)
    # loss_test = tf.reduce_mean(tf.square(y_pred - y_))

    # 训练模型
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(1000):
            # _, l = sess.run([train_op, loss], feed_dict={X: X_train, y: y_train})
            _, l = sess.run([train_op, loss])
            # if i % 10 == 0:
            #     print('Epoch {}, Loss: {}'.format(i, l))
        # l = sess.run(loss_test, feed_dict={X_: X_test, y_: y_test})
        l = sess.run(loss, feed_dict={X: X_test, y: y_test})
        print("test loss=", l)
    return l



def evaluate_models(X, y):
        # 进行数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("X_train=",X_train)
    # 对数据进行归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    l = train_and_evaluate(X_train, y_train, X_test, y_test)
    return l


# 假设你已经定义了X1, y1, X2, y2, X3, y3
# 评估三个数据集
# results_X1 = evaluate_models(X1, y1)

results_X3 = evaluate_models(X3, y3)
results_X2 = evaluate_models(X2, y2)




# 打印结果
# print("Results for X1: {}".format(results_X1))

print("\nResults for X2: {}".format(results_X2))

print("\nResults for X3: {}".format(results_X3))




# 比较三个数据集的结果
print("\nComparison:")

# best_dataset = min((results_X1, 'X1'), (results_X2, 'X2'), (results_X3, 'X3'), key=lambda x: x[0])
best_dataset = min((results_X2, 'X2'), (results_X3, 'X3'), key=lambda x: x[0])
print(f"dataset {best_dataset[1]} is better")