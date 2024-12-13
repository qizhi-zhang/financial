import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_1 = pd.read_csv('季度+年度数据-方案一.csv')
df_2 = pd.read_csv('年度数据-方案一.csv')


# 根据字段 'A' 的不同取值分组
# grouped = df_2.groupby('Symbol')

# # 遍历分组后的结果
# for name, group in grouped:
#     print(f"Group: {name}")
#     print(group)
#     group.to_csv('./dataset2/{}.csv'.format(int(name)), index=False)
    

def preprocess(df, save_path, party_num=3):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print("df_train=", df_train)
    print("df_test=", df_test)
    scaler = StandardScaler()
    df_train.iloc[:,:-1] = scaler.fit_transform(df_train.iloc[:,:-1])
    df_test.iloc[:,:-1] = scaler.transform(df_test.iloc[:,:-1])
    df_test.to_csv('{}/test.csv'.format(save_path), index=False)

    df_trains = np.array_split(df_train, party_num)
    # # 打印每个小 DataFrame
    for i, small_df in enumerate(df_trains):
        print(f"Small DataFrame {i}:")
        print(small_df)
        small_df.to_csv('{}/train_{}.csv'.format(save_path, int(i)), index=False)




# # #取出(年度+季度)数据的X和y
# # X1 = df_1.iloc[:,2:-1]
# # y1 = df_1.iloc[:,-1:]

# #取出年度数据的X和y
# X2 = df_2.iloc[:,2:-1]
# y2 = df_2.iloc[:,-1:]


# #取出季度数据的X和y
# X3 = df_1.iloc[:,15:-1]
# y3 = df_1.iloc[:,-1:]


preprocess(df_2.iloc[:,2:], "./dataset2_10party", 10)
preprocess(df_1.iloc[:,15:], "./dataset3_10party", 10)



