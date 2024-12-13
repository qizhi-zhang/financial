import pandas as pd
import numpy as np
from functools import reduce
np.random.seed(42)
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf
import argparse
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()




tf.compat.v1.disable_eager_execution()


def start_distributed_server(config, job_name, task_index):
    # config = json.json(config_file)
    cluster = tf.train.ClusterSpec(config)
    server = tf.distribute.Server(cluster, job_name=job_name, task_index=task_index)
    return server
    
def start_local_server(config):
    cluster = tf.train.ClusterSpec(config)
    tf.distribute.Server(cluster, job_name="ps", task_index=0)
    for task_index in range(len(config["worker"])):
        tf.distribute.Server(cluster, job_name="worker", task_index=task_index)



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
    # def __init__(self, input_dim):
    #     super(FCNN, self).__init__()
    #     self.w1 = tf.Variable(np.random.normal(size=[input_dim, 256])/input_dim+0.0, dtype='float32')
    #     self.b1 = tf.Variable(np.zeros([1,256]), dtype='float32')
    #     self.w2 = tf.Variable(np.random.normal(size=[256, 128])/(256+128.0), dtype='float32')
    #     self.b2 = tf.Variable(np.zeros([1, 128]), dtype='float32')
    #     self.w3 = tf.Variable(np.random.normal(size=[128, 64])/(128+64.0), dtype='float32')
    #     self.b3 = tf.Variable(np.zeros([1, 64]), dtype='float32')
    #     self.w4 = tf.Variable(np.random.normal(size=[64, 32])/(64+32.0), dtype='float32')
    #     self.b4 = tf.Variable(np.zeros([1, 32]), dtype='float32')
    #     self.w5 = tf.Variable(np.random.normal(size=[32, 1]), dtype='float32')
    #     self.b5 = tf.Variable(np.zeros([1, 1]), dtype='float32')

        
    
    def forward(self, x):
        with tf.name_scope("forward"):
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
        
        
        
        
def grads_and_vars_plus(list_of_truple1, list_of_truple2):
    r = []
    for i in range(len(list_of_truple1)):
        a1, b1 = list_of_truple1[i]
        a2, b2 = list_of_truple2[i]
        assert b1==b2
        r += [(a1+a2, b1)]
    return r

def grads_and_vars_div(list_of_truple, d):
    r = []
    for i in range(len(list_of_truple)):
        a, b = list_of_truple[i]
        r += [(a/d, b)]
    return r
         
    

def get_dataset(path, epoch, batch_size):
    with tf.name_scope("read_data"):
        data = tf.compat.v1.data.TextLineDataset([path]).skip(1).repeat(epoch+2).batch(batch_size).make_one_shot_iterator().get_next()
        data = tf.reshape(data, [batch_size])
        data = tf.strings.split(data, sep=",").to_tensor(default_value="0.0")
        data = tf.reshape(data, [batch_size, -1])
        data = tf.strings.to_number(data, out_type='float32')
        X = data[:,:-1]
        Y = data[:,-1]
        return X, Y



# 训练和评估模型的函数
def train_and_evaluate(train_datasets, test_datasets, input_dim, epoch, config):
    ps =  tf.DeviceSpec(job="ps", task=0)
    worker_num = len(config["worker"])
    print("worker_num=", worker_num)
    workers = list(map(lambda task_id: tf.DeviceSpec(job="worker", task=task_id), range(worker_num)))
    with tf.name_scope("ps"), tf.device(ps):
        fcnn = FCNN(input_dim)
    loss_list = []
    pred_loss_list = []
    grads_and_vars_list = []
    for i in range(worker_num):
        print("construct computing graph for worker {}".format(i))
        with tf.name_scope("worker_{}".format(i)), tf.device(workers[i]):
            batch_size=844
            X, y = get_dataset(train_datasets[i], epoch, batch_size)            
            y_pred = fcnn.forward(X)
            # print("y_pred.shape=", y_pred.shape)
            
            
            # 定义损失函数和优化器
            with tf.name_scope("train_loss"):
                loss = tf.reduce_mean(tf.square(y_pred - y))
                # loss = tf.math.log(loss)
            # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
            loss_list += [loss]
            grads_and_vars_list += [optimizer.compute_gradients(loss)]
            
            
            #-----------pred-------------------
            X, y = get_dataset(test_datasets[i%len(test_datasets)], epoch, test_datasets_size[i%len(test_datasets)])            
            y_pred = fcnn.forward(X)
            with tf.name_scope("test_loss"):
                pred_loss = tf.reduce_mean(tf.square(y_pred - y))
            pred_loss_list += [pred_loss]
    
    # print("grads_and_vars_list[0]=", grads_and_vars_list[0])
    with tf.name_scope("ps"), tf.device(ps):
        with tf.name_scope("grads_agg"):
            grads_and_vars = reduce(grads_and_vars_plus, grads_and_vars_list)
            grads_and_vars = grads_and_vars_div(grads_and_vars, worker_num)
        with tf.name_scope("train_loss_agg"):
            loss = sum(loss_list)
            loss = loss / worker_num
        train_op = optimizer.apply_gradients(grads_and_vars)
        with tf.name_scope("pred_loss_agg"):
            pred_loss = sum(pred_loss_list)
            pred_loss = pred_loss / worker_num
        

    
    # 训练模型
    with tf.compat.v1.Session("grpc://127.0.0.1:6540") as sess:
        train_loss_summary = tf.compat.v1.summary.scalar("test loss", pred_loss)
        writer = tf.compat.v1.summary.FileWriter("./tfboard/", sess.graph)
        summaries = tf.compat.v1.summary.merge_all()
        
        sess.run(tf.compat.v1.global_variables_initializer())
        
        
        # sess.run(summary_writer)
        for i in range(epoch):
            # _, l = sess.run([train_op, loss], feed_dict={X: X_train, y: y_train})
            _, l, summ = sess.run([train_op, pred_loss, summaries])
            if i % 5 == 0:
                print('Epoch {}, Loss: {}'.format(i, l))    
                writer.add_summary(summ, global_step=i)
        # l = sess.run(pred_loss, feed_dict={X: X_test, y: y_test})
        l = sess.run(pred_loss)
        # print("test loss=", l)
    return l





if __name__ == '__main__':

    # 添加命令行参数
    parser.add_argument('--local', '-l', type=int, default=1, help='local or distribute')
    parser.add_argument('--job_name', '-j', type=str, choices=["ps", "worker"])
    parser.add_argument('--task_index', '-t', type=int)
    # 解析命令行参数
    args = parser.parse_args()


    feature_num = 43
    train_datasets = ["/home/zhangqizhi.zqz/projects/financial/stock/dataset3_10party/train_{}.csv".format(i) for i in range(10)]

    test_datasets = ["/home/zhangqizhi.zqz/projects/financial/stock/dataset3_10party/test.csv"]
    test_datasets_size = [2110]

    config = {"ps": ["127.0.0.1:6540"],
            "worker": ["127.0.0.1:{}".format(i) for i in range(6541, 6541+len(train_datasets))]}
    epoch = 100
    if args.local==1:
        start_local_server(config)
        results_loss = train_and_evaluate(train_datasets, test_datasets, feature_num, epoch, config)
        print("使用年度数据训练, 预测方差={}".format(results_loss))
    else:
        server = start_distributed_server(config, args.job_name, args.task_index)
        if args.job_name=="ps":
            results_loss = train_and_evaluate(train_datasets, test_datasets, feature_num, epoch, config)
            print("使用季度数据训练, 预测方差={}".format(results_loss))
        else:
            server.join()



