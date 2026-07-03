# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc,expand_neighbors
from util.iostream import printsep
from util.find_root_cause_node import find_root_cause_node
from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import json
import random
from util.iostream import  save_attack_infos


class MinMaxScaler:
    def fit(self, data):
        arr = np.asarray(data, dtype=float)
        self.data_min_ = np.min(arr, axis=0)
        self.data_max_ = np.max(arr, axis=0)
        data_range = self.data_max_ - self.data_min_
        self.scale_ = np.where(data_range == 0, 1.0, data_range)
        return self

    def transform(self, data):
        arr = np.asarray(data, dtype=float)
        return (arr - self.data_min_) / self.scale_

    def fit_transform(self, data):
        return self.fit(data).transform(data)



class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        self.dataset=env_config['dataset']
        train_orig = pd.read_csv(f'./data/{dataset}/HAI_train.csv', sep=',', index_col=0)#读取csv文件（原始数据），将第一列作为行索引
        test_orig = pd.read_csv(f'./data/{dataset}/HAI_test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])#如果有colums里边有attack，就删除attack那一列

        feature_map = get_feature_map(dataset)#得到points列表
        fc_struc = get_fc_graph_struc(dataset)#得到带有邻居名字并含有边权重的points字典(未扩展的)
        expended_fc_struc=expand_neighbors(fc_struc,4,feature_map=feature_map)
        fc_struc = get_fc_graph_struc(
            dataset,
            make_undirected=env_config.get('prior_undirected', True),
            verbose=env_config.get('prior_verbose', False)
        )
        expended_fc_struc = expand_neighbors(
            fc_struc,
            train_config.get('prior_expand', 4),
            feature_map=feature_map,
            min_neighbors=train_config.get('prior_min_neighbors', 0),
            verbose=env_config.get('prior_verbose', False)
        )
        self.expended_fc_struc=expended_fc_struc#得到拓扑增强后的图结构

        set_device(env_config['device'])
        self.device = get_device()#cpu or gpu训练

        fc_edge_index = build_loc_net(expended_fc_struc, list(train.columns), feature_map=feature_map)#构建边矩阵（2行多列）
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)#变为tensor张量，数据类型为torch.long

        self.feature_map = feature_map

        if train_config.get('normalize', True):
            scaler = MinMaxScaler()
            scale_features = [ft for ft in feature_map if ft in train.columns and ft in test.columns]
            train.loc[:, scale_features] = scaler.fit_transform(train.loc[:, scale_features])
            test.loc[:, scale_features] = scaler.transform(test.loc[:, scale_features])

        train_dataset_indata = construct_data(train, feature_map, labels=0,return_sample_n=False)#得到 最后一列是attack（attack初始值都为0）的 具有所有points特征（时间）的 二维列表 作为train_dataset
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist(),return_sample_n=False)
        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
            'result_save_path' : env_config['result_save_path']
        }
        self.slide_win = cfg['slide_win']
        self.result_save_path = cfg['result_save_path']
        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)#滑动加工，凸显时间特性
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)


        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)#返回  可迭代 的数据集


        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)#得到边的索引（边的信息）

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk'],
                graph_mode=train_config.get('graph_mode', 'prior_mask')
            ).to(self.device)#传入参数，初始化GDN



    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            #start train
            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)#返回损失，预测值 真实值 以及标签

        f1_scores,total_err_scores,labels=self.get_score(self.test_result, self.val_result,return_value=True)

        # self.save_attack_infos_and_find_father_node(f1_scores,total_err_scores,labels,names=self.feature_map)
    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):#val_ratio为验证集占比,抽离出val dataset
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))#train_dataset长度
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)#生成一个随机数
        indices = torch.arange(dataset_len)#创建一个一维tensor

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])#拼接
        train_subset = Subset(train_dataset, train_sub_indices)#相减，得到需要使用的dataset

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result,return_value=False):#评估分数

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()#np_test_result是一个array
    
        test_scores, normal_scores ,labels= get_full_err_scores(test_result, val_result)
        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) #得到最大的f1 score
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)#验证集的表现


        print('=========================** Result **============================\n')
        #训练结果部分，打印evaluae参数
        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')
        if return_value:
            return  info,test_scores, labels


    def get_save_path(self, feature_name=''):#保存路径

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m-%d-%H-%M-%S')#现在的时间
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths
    # def save_attack_infos_and_find_father_node(self,f1_scores, total_err_scores, labels, names,):
    #     #save attack infos
    #     time_period_max_sensor=save_attack_infos(f1_scores, total_err_scores, labels, names, self.result_save_path, self.dataset, self.slide_win,down_len=10)
    #     #find father node(attack propogation chain)
    #     # 先转置 total_errs_score，以便每一列对应一个传感器的时序数据
    #     total_err_scores_transposed = total_err_scores.T
    #     # 创建一个 DataFrame，使用转置后的数据，同时使用 feature_map 作为列索引
    #     total_err_scores_pd = pd.DataFrame(total_err_scores_transposed, index=range(25047), columns=self.feature_map)
    #
    #     # 现在通过传感器名字来索引列，获取对应的时序数据
    #     sensor1_data = total_err_scores_pd.loc[:, 'DM-FT01Z']
    #
    #     find_root_cause_node(self.expended_fc_struc,total_err_scores_pd,time_period_max_sensor,slide_win=self.slide_win,down_len=10)

if __name__ == "__main__":#传入的参数

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=16)#源代码为128，但论文中未提供这个参数数值
    parser.add_argument('-epoch', help='train epoch', type = int, default=50)#
    parser.add_argument('-slide_win', help='slide_win', type = int, default=5)#
    parser.add_argument('-dim', help='dimension', type = int, default=64)#embedding_dim 源代码为64 论文为128/64(wadi/swat)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=1)#论文未提供这个参数数值
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='hai23.05 / hai23.05_end', type = str, default='hai23.05_end')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=6799)#random seed########################################
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=128)#128/64 (wadi/swat)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=15)#该数据集邻居较少
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-graph_mode', help='learned / prior / prior_mask', type = str, default='prior_mask')
    parser.add_argument('-prior_expand', help='prior graph expansion hops', type = int, default=4)
    parser.add_argument('-prior_min_neighbors', help='fallback minimum neighbors, 0 keeps only prior edges', type = int, default=0)
    parser.add_argument('-prior_undirected', help='use undirected physical prior candidates', type = int, default=1)
    parser.add_argument('-normalize', help='fit MinMaxScaler on train and transform train/test', type = int, default=1)
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-result_save_path', help='result save path', type=str, default=r'G:\GDN-main-fuben\results\result.txt')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'graph_mode': args.graph_mode,
        'prior_expand': args.prior_expand,
        'prior_min_neighbors': args.prior_min_neighbors,
        'normalize': bool(args.normalize),
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'result_save_path': args.result_save_path,
        'prior_undirected': bool(args.prior_undirected),
        'prior_verbose': False,
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()
