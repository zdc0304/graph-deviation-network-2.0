# preprocess data
import numpy as np
import re
import pandas as pd
import numpy as np
#preprocess 预加工处理
def get_most_common_features(target, all_features, max = 3, min = 3):#只在该文件中使用了一次
    res = []
    main_keys = target.split('_')#返回按照‘_’分割的列表

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))#计算f_keys和main_keys的共同元素的个数

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res

def build_net(target, all_features):#该函数在all place中没有被发现使用
    # get edge_indexes, and index_feature_map
    main_keys = target.split('_')
    edge_indexes = [
        [],
        []
    ]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2
    
    for i in range(depth):        
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []
            
            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map


def construct_data(data, feature_map, labels=0,return_sample_n=False):#构建数据集 对|||||原始数据进行归一化||||||||||||||||||||||||||||||||||||||||


    # # 假设原始数据保存在DataFrame中，索引为时间戳，列为特征
    # # 假设原始数据采样频率为每秒一次
    #
    # # 将索引转换为DatetimeIndex类型
    # data.index = pd.to_datetime(data.index)
    # # 重采样为每十秒取平均值
    # resampled_data = data.resample('10S').mean()
    # # 处理异常值
    # resampled_data.fillna(0, inplace=True)  # 填充NaN值为0
    ## Min-Max Scaling

    # res = []  # 用于存储归一化后的数据
    # for feature in feature_map:
    #     if feature in data.columns:
    #         feature_array = np.array(data[feature])
    #         data_feature_min = np.min(feature_array)
    #         data_feature_max = np.max(feature_array)
    #         if data_feature_max == data_feature_min:  # 如果最大值和最小值相等0
    #             scaled_data = np.zeros_like(feature_array)
    #         else:
    #             scaled_data = (feature_array - data_feature_min) / (data_feature_max - data_feature_min)
    #         res.append(scaled_data.tolist())  # 转变为列表，把feature加入res列表中
    #     else:
    #         print(feature, 'not exist in data')

    res = []
    # feature在columns里边，就添加到res列表里边
    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())  # 先转变为数组，再变为列表，把feature加入res列表中
        else:
            print(feature, 'not exist in data')
    # append labels as last，将标签放在最后边
    sample_n = len(res[0])#采样持续时间
    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)
    if return_sample_n:
        return res, sample_n
    else:
        return res

def build_loc_net(struc, all_features, feature_map=[]):#构建局部网络的边信息
# learns the relationships between pairs of sensors, and encodes them as edges in a graph;
    index_feature_map = feature_map#以points名字作为邻居网络字典的索引
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)#构建points名字的列表
        
        p_index = index_feature_map.index(node_name)#以名字作为索引
        for child in node_list:#子节点（邻居）
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)#邻居在feature_map里的索引
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)#邻居索引
            edge_indexes[1].append(p_index)#自身索引
        

    return edge_indexes