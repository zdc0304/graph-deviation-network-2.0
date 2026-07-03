import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import numpy as np

#将dataset再处理加工一遍，3.5Thus, at time t, we define our model input x(t) ∈ RN×w
# based on a sliding window of size w over the historical time
# series data (whether training or testing data):（4）

class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index#边索引
        self.mode = mode

        x_data = raw_data[:-1]#去除attack，x_data具有所有传感器特征
        labels = raw_data[-1]#将attack作为标签


        data = x_data

        # to tensor and dtype=double,类似转置
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr = [], []#array
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]#传入滑动大小和步长
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape#返回传感器数量、时间点的长度

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)#slide_win滑动大小，slide_stride滑动步长
        
        for i in rang:

            ft = data[:, i-slide_win:i]#所有传感器在这段时间的特征
            tar = data[:, i]#所有传感器在i时间点的特征

            x_arr.append(ft)#构建由张量ft组成的array
            y_arr.append(tar)

            labels_arr.append(labels[i])#所有传感器在i时间点的真实标签


        x = torch.stack(x_arr).contiguous()#沿新维度堆叠，在这行代码中实则是将x_arr变为tensor
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()#所有传感器在第x个时间段的特征
        y = self.y[idx].double()#所有传感器在第x个时间点的特征

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index




