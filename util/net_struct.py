import glob
import json
from collections import Counter
#得到net_struct，包括传感器名字列表，以及传感器图
def get_feature_map(dataset):#打开相应数据集中的list文件并保存为列表形式，得到传感器名字列表，不具有特征信息
    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())#.strip()可以用以删除空格

    return feature_list
# graph is 'fully-connect'
def get_fc_graph_struc(dataset):#得到传感器图
    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    neighbor_struc_file=[f'./data/{dataset}/dcs_1001h.json',f'./data/{dataset}/dcs_1002h.json',f'./data/{dataset}/dcs_1003h.json',f'./data/{dataset}/dcs_1004h.json',f'./data/{dataset}/dcs_1010h.json',f'./data/{dataset}/dcs_1011h.json',f'./data/{dataset}/dcs_1020h.json']
    feature_list = []
    # 创建一个空字典用于存储结果
    neighbor_results= {}
    for ft in feature_file:
        feature_list.append(ft.strip())#得到具有所有传感器名字的列表
    # 循环遍历文件列表
    for filename in neighbor_struc_file:
        # 打开文件
        with open(filename, 'r') as file:
            # 读取文件内容
            data = json.load(file)
        for link1 in data["links"]:
            source1 = link1["source"]
            target1 = link1["target"]
            label1 = link1["label"]

            for link2 in data["links"]:
                source2 = link2["source"]
                target2 = link2["target"]
                label2 = link2["label"]

                if target1 == source2:
                    if label2 not in neighbor_results :
                        neighbor_results[label2] = [label1]
                    elif label2 in neighbor_results :
                        neighbor_results[label2].append(label1)
    neighbor_results= {key: [value for value in values if value != key] for key, values in neighbor_results.items()}#删除自环边

    for ft in feature_list:

        if ft not in neighbor_results:
            neighbor_results[ft]=[]

        # if neighbor_results[ft]==[]:
        #     for other_ft in feature_list:
        #         if other_ft==ft:
        #             continue
        #         else:
        #             neighbor_results[ft].append(other_ft)

    print(neighbor_results)

    # 合并重复points并计算权重
    # for key in neighbor_results:
    #     neighbor_results[key] = dict(Counter(neighbor_results[key]))
    return neighbor_results
def expand_neighbors(neighbor_results, expansion_level,feature_map):
    expanded_neighbors = {}  # 用于存储扩展后的邻居关系
    for sensor, neighbors in neighbor_results.items():
        current_neighbors = set(neighbors)
        if expansion_level > 1:
            # 如果指定了扩展层数大于1，则进行递归扩展
            for level in range(2, expansion_level + 1):
                new_neighbors = set()
                for neighbor in current_neighbors:
                    if neighbor in neighbor_results:  # 确保邻居在原始邻居字典中存在
                        new_neighbors.update(neighbor_results[neighbor])
                current_neighbors.update(new_neighbors)
        expanded_neighbors[sensor] = list(current_neighbors)
    # 扩展工作完成后删除不在 feature_map 中的邻居,以及sensor
    expanded_neighbors = {
        sensor: [neighbor for neighbor in neighbors if neighbor in feature_map]
        for sensor, neighbors in expanded_neighbors.items()
        if sensor in feature_map
    }
    print(expanded_neighbors)
    # 打印调试信息，确认每个传感器的邻居数量
    for sensor, neighbors in expanded_neighbors.items():
        print(f"Sensor: {sensor}, Number of Neighbors: {len(neighbors)}")

    # 遍历 expanded_neighbors，确保每个传感器的邻居数量至少为 15
    for sensor, neighbors in expanded_neighbors.items():
        if len(neighbors) < 15:
            # 从 feature_map 中添加其他节点，直到邻居数量达到 15
            additional_neighbors = [
                node for node in feature_map if node != sensor and node not in neighbors
            ]
            neighbors.extend(additional_neighbors[:15 - len(neighbors)])

    # 打印调试信息，确认每个传感器的邻居数量
    for sensor, neighbors in expanded_neighbors.items():
        print(f"Sensor: {sensor}, Number of Neighbors: {len(neighbors)}")

    # 确保 expanded_neighbors 更新后的结构是字典
    expanded_neighbors = {
        sensor: neighbors for sensor, neighbors in expanded_neighbors.items()
    }





    return expanded_neighbors

def get_prior_graph_struc(dataset):#无feature.txt文件，所以未被使用
    feature_file = open(f'./data/{dataset}/features.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if dataset == 'wadi' or dataset == 'wadi2':
                # same group, 1_xxx, 2A_xxx, 2_xxx
                if other_ft is not ft and other_ft[0] == ft[0]:
                    struc_map[ft].append(other_ft)
            elif dataset == 'swat':
                # FIT101, PV101
                if other_ft is not ft and other_ft[-3] == ft[-3]:
                    struc_map[ft].append(other_ft)

    
    return struc_map


def get_fc_graph_struc(dataset, make_undirected=False, verbose=False):
    feature_path = f'./data/{dataset}/list.txt'
    neighbor_struc_file = [
        f'./data/{dataset}/dcs_1001h.json',
        f'./data/{dataset}/dcs_1002h.json',
        f'./data/{dataset}/dcs_1003h.json',
        f'./data/{dataset}/dcs_1004h.json',
        f'./data/{dataset}/dcs_1010h.json',
        f'./data/{dataset}/dcs_1011h.json',
        f'./data/{dataset}/dcs_1020h.json',
    ]

    with open(feature_path, 'r') as feature_file:
        feature_list = [ft.strip() for ft in feature_file if ft.strip()]
    feature_set = set(feature_list)

    neighbor_results = {ft: [] for ft in feature_list}
    for filename in neighbor_struc_file:
        with open(filename, 'r') as file:
            graph_data = json.load(file)

        links = graph_data["links"]
        for link1 in links:
            for link2 in links:
                if link1["target"] == link2["source"]:
                    source_label = link1["label"]
                    target_label = link2["label"]
                    if source_label == target_label:
                        continue
                    if target_label in feature_set and source_label in feature_set:
                        neighbor_results.setdefault(target_label, []).append(source_label)

    for sensor, neighbors in list(neighbor_results.items()):
        deduped = []
        seen = set()
        for neighbor in neighbors:
            if neighbor == sensor or neighbor in seen:
                continue
            deduped.append(neighbor)
            seen.add(neighbor)
        neighbor_results[sensor] = deduped

    if make_undirected:
        for sensor, neighbors in list(neighbor_results.items()):
            for neighbor in list(neighbors):
                if neighbor not in neighbor_results:
                    continue
                if sensor not in neighbor_results[neighbor] and sensor != neighbor:
                    neighbor_results[neighbor].append(sensor)

    if verbose:
        print(neighbor_results)

    return neighbor_results


def expand_neighbors(neighbor_results, expansion_level, feature_map, min_neighbors=0, verbose=False):
    feature_set = set(feature_map)
    expanded_neighbors = {}

    for sensor, neighbors in neighbor_results.items():
        current_neighbors = set(neighbors)
        for _ in range(2, expansion_level + 1):
            new_neighbors = set()
            for neighbor in current_neighbors:
                if neighbor in neighbor_results:
                    new_neighbors.update(neighbor_results[neighbor])
            current_neighbors.update(new_neighbors)

        filtered = [neighbor for neighbor in current_neighbors if neighbor in feature_set and neighbor != sensor]
        expanded_neighbors[sensor] = filtered

    expanded_neighbors = {
        sensor: neighbors
        for sensor, neighbors in expanded_neighbors.items()
        if sensor in feature_set
    }

    if min_neighbors > 0:
        for sensor, neighbors in expanded_neighbors.items():
            if len(neighbors) >= min_neighbors:
                continue
            additional_neighbors = [
                node for node in feature_map if node != sensor and node not in neighbors
            ]
            neighbors.extend(additional_neighbors[:min_neighbors - len(neighbors)])

    if verbose:
        for sensor, neighbors in expanded_neighbors.items():
            print(f"Sensor: {sensor}, Number of Neighbors: {len(neighbors)}")

    return expanded_neighbors


if __name__ == '__main__':
    get_graph_struc()
 
