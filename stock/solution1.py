import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


df_1 = pd.read_csv('季度+年度数据-方案一.csv')
df_2 = pd.read_csv('年度数据-方案一.csv')

df_1.head()


#取出(年度+季度)数据的X和y
X1 = df_1.iloc[:,2:-1]
y1 = df_1.iloc[:,-1]

#取出年度数据的X和y
X2 = df_2.iloc[:,2:-1]
y2 = df_2.iloc[:,-1]

#取出季度数据的X和y
X3 = df_1.iloc[:,15:-1]
y3 = df_1.iloc[:,-1]


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# 定义全连接神经网络
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * input_dim, 256)  # 动态调整输入长度
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 定义长短期记忆网络
class LSTM(nn.Module):
    def __init__(self, input_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, 100, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 100)
        c_0 = torch.zeros(2, x.size(0), 100)
        out, _ = self.lstm(x, (h_0, c_0))
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out

# 定义回归算法
regressors = {
    'Fully Connected Neural Network': FCNN,
    # 'Convolutional Neural Network': CNN,
    # 'Long Short-Term Memory': LSTM
}

# 训练和评估模型的函数
def train_and_evaluate(model_class, X_train, y_train, X_test, y_test, input_dim):
    model = model_class(input_dim)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    if model_class == CNN:
        X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), 1, -1)
        X_test_tensor = X_test_tensor.view(X_test_tensor.size(0), 1, -1)
    elif model_class == LSTM:
        X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1, input_dim)
        X_test_tensor = X_test_tensor.view(X_test_tensor.size(0), -1, input_dim)
    
    for epoch in range(100):  # 训练100个epoch
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
    
    mse = mean_squared_error(y_test, predictions)
    return mse

# 函数来评估每个数据集的精度
def evaluate_models(X, y):
    results = {}
    
    # 进行数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # 对数据进行归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    input_dim = X_train.shape[1]
    
    for name, model_class in regressors.items():
        mse = train_and_evaluate(model_class, X_train, y_train, X_test, y_test, input_dim)
        results[name] = mse
    return results

# 假设你已经定义了X1, y1, X2, y2, X3, y3
# 评估三个数据集
results_X1 = evaluate_models(X1, y1)
results_X2 = evaluate_models(X2, y2)
results_X3 = evaluate_models(X3, y3)

# 打印结果
print("Results for X1:")
for name, mse in results_X1.items():
    print(f"{name}: MSE = {mse}")

print("\nResults for X2:")
for name, mse in results_X2.items():
    print(f"{name}: MSE = {mse}")

print("\nResults for X3:")
for name, mse in results_X3.items():
    print(f"{name}: MSE = {mse}")

# 比较三个数据集的结果
print("\nComparison:")
for name in regressors.keys():
    best_dataset = min((results_X1[name], 'X1'), (results_X2[name], 'X2'), (results_X3[name], 'X3'), key=lambda x: x[0])
    print(f"{name}: {best_dataset[1]} is better")