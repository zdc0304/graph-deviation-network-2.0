import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1) scaling
def norm(train, test):#将train1_dataset and test_dataset归一化

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale train1ing data to [0,1] range
    train1_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train1_ret, test_ret

# downsample by 10 10s为一组采一次样，再取平均
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape #行数（时序数据长度）  列数（传感器个数+label数）

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()
    # print('before downsample', np_data.shape)

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))

    d_data = d_data.transpose()

    # print('after downsample', d_data.shape, d_labels.shape)

    return d_data.tolist(), d_labels.tolist()


def main():

    # 加载数据集
    train1 = pd.read_csv('./end-train1.csv', index_col=0)
    train2 = pd.read_csv('./end-train2.csv', index_col=0)
    test1 = pd.read_csv('./end-test1.csv', index_col=0)
    test2 = pd.read_csv('./end-test2.csv', index_col=0)
    train3 = pd.read_csv('./end-train3.csv', index_col=0)
    train4 = pd.read_csv('./end-train4.csv', index_col=0)

    # 去除第一列（时间列）
    train1 = train1.iloc[:, 1:]
    train2 = train2.iloc[:, 1:]
    test1 = test1.iloc[:, 1:]
    test2 = test2.iloc[:, 1:]
    train3 = train3.iloc[:, 1:]
    train4 = train4.iloc[:, 1:]
    # 填充缺失值
    train1 = train1.fillna(train1.mean())
    train2 = train2.fillna(train2.mean())
    test1 = test1.fillna(test1.mean())
    test2 = test2.fillna(test2.mean())
    train3 = train3.fillna(train3.mean())
    train4 = train4.fillna(train4.mean())
    # 去除缺失值
    train1 = train1.fillna(0)
    train2 = train2.fillna(0)
    test1 = test1.fillna(0)
    test2 = test2.fillna(0)
    train3 = train3.fillna(0)
    train4 = train4.fillna(0)
    # 去除列名中的空格
    train1 = train1.rename(columns=lambda x: x.strip())
    train2 = train2.rename(columns=lambda x: x.strip())
    test1 = test1.rename(columns=lambda x: x.strip())
    test2 = test2.rename(columns=lambda x: x.strip())
    train3 = train3.rename(columns=lambda x: x.strip())
    train4 = train4.rename(columns=lambda x: x.strip())
    # 提取标签列
    train1_labels = np.zeros(len(train1))
    train2_labels = np.zeros(len(train2))
    test1_labels = test1.attack
    test2_labels = test2.attack
    train3_labels = np.zeros(len(train3))
    train4_labels = np.zeros(len(train4))
    # 去除标签列
    if 'attack' in train1.columns:
        train1 = train1.drop(columns=['attack'])
    if 'attack' in train2.columns:
        train2 = train2.drop(columns=['attack'])
    if 'attack' in train3.columns:
        train3 = train3.drop(columns=['attack'])
    if 'attack' in train4.columns:
        train4 = train4.drop(columns=['attack'])
    test1 = test1.drop(columns=['attack'])
    test2 = test2.drop(columns=['attack'])

    # 归一化数据
    x_train1, x_test1 = norm(train1.values, test1.values)
    x_train2, x_test2 = norm(train2.values, test2.values)
    x_train3, x_train4 = norm(train3.values, train4.values)
    # 对数据进行赋值
    for i, col in enumerate(train1.columns):
        train1.loc[:, col] = x_train1[:, i]
        train2.loc[:, col] = x_train2[:, i]
        test1.loc[:, col] = x_test1[:, i]
        test2.loc[:, col] = x_test2[:, i]
        train3.loc[:, col] = x_train3[:, i]
        train4.loc[:, col] = x_train4[:, i]

    # 下采样
    d_train1_x, d_train1_labels = downsample(train1.values, train1_labels, 10)
    d_train2_x, d_train2_labels = downsample(train2.values, train2_labels, 10)
    d_test1_x, d_test1_labels = downsample(test1.values, test1_labels, 10)
    d_test2_x, d_test2_labels = downsample(test2.values, test2_labels, 10)
    d_train3_x, d_train3_labels = downsample(train3.values, train3_labels, 10)
    d_train4_x, d_train4_labels = downsample(train4.values, train4_labels, 10)

    # 创建DataFrame
    train1_df = pd.DataFrame(d_train1_x, columns=train1.columns)
    train2_df = pd.DataFrame(d_train2_x, columns=train2.columns)
    test1_df = pd.DataFrame(d_test1_x, columns=test1.columns)
    test2_df = pd.DataFrame(d_test2_x, columns=test2.columns)
    train3_df = pd.DataFrame(d_train3_x, columns=train3.columns)
    train4_df = pd.DataFrame(d_train4_x, columns=train4.columns)
    # 添加标签列
    test1_df['attack'] = d_test1_labels
    test2_df['attack'] = d_test2_labels
    train1_df['attack'] = d_train1_labels
    train2_df['attack'] = d_train2_labels
    train3_df['attack'] = d_train3_labels
    train4_df['attack'] = d_train4_labels
    # 合并测试数据集
    test_df = pd.concat([test1_df, test2_df], ignore_index=True)


    # 丢弃前2160秒的数据
    train1_df = train1_df.iloc[2160:]
    train2_df = train2_df.iloc[2160:]
    train3_df = train3_df.iloc[2160:]
    train4_df = train4_df.iloc[2160:]


    # 合并训练数据集
    train_df = pd.concat([train1_df, train2_df,train3_df,train4_df], ignore_index=True)


    columns_to_check = train_df.columns[0:-1]
    # 遍历这些列
    for column in columns_to_check:
        # 检查该列是否全是 0
        if (train_df[column] == 0).all() and (test_df[column] == 0).all():
            # 删除 train_df 和 test_df 中的该列
            train_df.drop(column, axis=1, inplace=True)
            test_df.drop(column, axis=1, inplace=True)
    # 保存数据

    with open('./list.txt', 'w') as f:
        for col in train_df.columns[0:-1]:
            f.write(col + '\n')
    train_df.to_csv('./HAI_train.csv')
    test_df.to_csv('./HAI_test.csv')
    f.close()  # 保存具有所有传感器名字的文件

if __name__ == '__main__':
    main()
