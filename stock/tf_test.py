import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

# 创建训练数据
np.random.seed(0)
X_train = np.random.rand(100, 1)
y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1)

# 创建Tensorflow模型
X = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

W = tf.Variable(tf.random.normal([1, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

y_pred = tf.matmul(X, W) + b

# 定义损失函数和优化器
loss = tf.reduce_mean(tf.square(y_pred - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

# 训练模型
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(1000):
        _, l = sess.run([train_op, loss], feed_dict={X: X_train, y: y_train})
        if i % 100 == 0:
            print('Epoch {}, Loss: {}'.format(i, l))

    W_final, b_final = sess.run([W, b])
    print("训练结束，模型参数：")
    print("W: {}, b: {}".format(W_final[0][0], b_final[0]))