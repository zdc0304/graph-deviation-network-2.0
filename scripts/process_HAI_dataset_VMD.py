import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据
train = pd.read_csv('./HAI_train_VMD.csv', index_col=0)
test = pd.read_csv('./HAI_test_VMD.csv', index_col=0)

# 提取时间列
train_time = train.index
test_time = test.index

# 提取标签列
train_labels = np.zeros(len(train))
test_labels = test.attack

# 去除标签列
train = train.drop(columns=['attack'], errors='ignore')
test = test.drop(columns=['attack'], errors='ignore')

# 归一化
normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train)  # 使用去掉标签列后的训练数据进行fit
train_normalized = normalizer.transform(train)
test_normalized = normalizer.transform(test)

# 重新赋值给数据框
train_df = pd.DataFrame(train_normalized, columns=train.columns, index=train_time)
test_df = pd.DataFrame(test_normalized, columns=test.columns, index=test_time)

# 添加标签列
train_df['attack'] = train_labels
test_df['attack'] = test_labels

# 保存
train_df.to_csv('./HAI_train_VMD_scale.csv')
test_df.to_csv('./HAI_test_VMD_scale.csv')

# 检查时间列是否存在
print("Train file contains time column:", 'index' in train_df.columns)
print("Test file contains time column:", 'index' in test_df.columns)