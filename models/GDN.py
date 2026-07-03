import numpy as np
import torch
import torch.nn as nn
import time
from util.time import *
from util.env import *
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):#得到分批次的边索引,clone得到多张相同结构的图，以便在一个epoch中同时训练多张图，提高训练效率
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num#确保每张图的节点索引不重叠

    return batch_edge_index.long()#转为long结构


class OutLayer(nn.Module):#输出层
    def __init__(self, in_num, node_num, layer_num, inter_num = 128):#inter_layer_neurons
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:#最后一层添加这样的一个线性层，Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
                    modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))#若输出层只有一层，那么线性层的输入的列数为in_num，否则为inter_num，输出列数为1
            else:
                layer_in_num = in_num if i == 0 else inter_num#输出层的第一层添加nn.Linear( in_num, inter_num )和减少协变量的偏移以加快网络训练速度的batchNorm1d层
                modules.append(nn.Linear( layer_in_num, inter_num ))#除第一层和最后一层添加nn.Linear( inter_num, inter_num )，和batchNor1d
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())#添加激活层，避免过拟合

        self.mlp = nn.ModuleList(modules)#将子模块储存到列表当中

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):#如果mod是batchnorm1d层，则先对输入进行维度变，再传入mod层
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):#GNN层
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)#batch_norm1d层
        self.relu = nn.ReLU()#relu层
        self.leaky_relu = nn.LeakyReLU()#而非rulu层，负数不会完全归零

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=128, out_layer_inter_dim=128,
                 input_dim=5, out_layer_num=1, topk=10, graph_mode='prior_mask'):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets#边索引

        device = get_device()#判断使用cpu还是gpu训练

        edge_index = edge_index_sets[0]


        embed_dim = dim#3.3sensor embedding部分
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)#对输入数据做一个normalization


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])#定义的gnn层


        self.node_embedding = None
        self.topk = topk
        self.graph_mode = graph_mode
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)#输出层

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)#随机置0，This has proven to be an effective technique for regularization and
                                         # preventing the co-adaptation of neurons as described in the paper
                                        #`Improving neural networks by preventing co-adaptation of feature
                                        # detectors`_ .

        self.init_params()
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def get_prior_masked_edge_index(self, cos_ji_mat, edge_index, node_num, device):
        candidate_mask = torch.zeros((node_num, node_num), dtype=torch.bool, device=device)
        prior_edge_index = edge_index.long().to(device)

        if prior_edge_index.numel() > 0:
            sources = prior_edge_index[0]
            targets = prior_edge_index[1]
            valid = (sources >= 0) & (sources < node_num) & (targets >= 0) & (targets < node_num)
            candidate_mask[targets[valid], sources[valid]] = True

        node_ids = torch.arange(node_num, device=device)
        candidate_mask[node_ids, node_ids] = True

        source_parts = []
        target_parts = []
        for target_idx in range(node_num):
            candidates = torch.nonzero(candidate_mask[target_idx], as_tuple=False).view(-1)
            k = min(self.topk, candidates.numel())
            scores = cos_ji_mat[target_idx, candidates]
            selected = candidates[torch.topk(scores, k, dim=-1)[1]]
            source_parts.append(selected)
            target_parts.append(torch.full((k,), target_idx, dtype=torch.long, device=device))

        return torch.stack((torch.cat(source_parts), torch.cat(target_parts)), dim=0)

    def get_learned_edge_index(self, cos_ji_mat, node_num, device):
        topk_num = min(self.topk, node_num)
        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
        gated_i = torch.arange(0, node_num, device=device).unsqueeze(1).repeat(1, topk_num).flatten().unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        return torch.cat((gated_j, gated_i), dim=0)

    def merge_edge_index(self, edge_index_a, edge_index_b, node_num, device):
        merged = torch.cat((edge_index_a.long().to(device), edge_index_b.long().to(device)), dim=1)
        edge_ids = merged[0] * node_num + merged[1]
        unique_ids = torch.unique(edge_ids)
        return torch.stack((unique_ids // node_num, unique_ids % node_num), dim=0).long()


    def forward(self, data, org_edge_index):

        x = data.clone().detach()#输入数据clone 克隆 再 detach 分批次
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()


        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):#返回边索引的索引和边索引 enumerate用于返回索引和对应元素值
            edge_num = edge_index.shape[1]#tensor edge的列数表示有多少个边
            cache_edge_index = self.cache_edge_index_sets[i]#多张图的边索引

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:#是空列表或 者 列数不等于一张图的边数×批次数
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)#重新得到分批边索引
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))#node embedding；device is cpu or gpu

            weights_arr = all_embeddings.detach().clone()#切断weights_arr和图的连接 不参与梯度计算 和原始张量具有相同数值
            all_embeddings = all_embeddings.repeat(batch_num, 1)#重复  因为我们要同时跑多张图的数据 所以重复的数量是batch_num

            weights = weights_arr.view(node_num, -1)#行为传感器数量
            #3.4 graph strcture learning That is, we first compute eji,
            # the normalized dot product between the embedding vectors of sensor i,
            # and the candidaterelation j ∈ Ci
            # eji = （vi.T vj ）/（|vi| |vj|）  for j ∈ Ci (向量之间的cosθ)
            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            if self.graph_mode == 'prior_union':
                learned_edge_index = self.get_learned_edge_index(cos_ji_mat, node_num, device)
                gated_edge_index = self.merge_edge_index(learned_edge_index, edge_index, node_num, device)
                self.learned_graph = gated_edge_index.detach().cpu()
                batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
                gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
                gcn_outs.append(gcn_out)
                continue

            if self.graph_mode == 'prior':
                gated_edge_index = edge_index.long().to(device)
                self.learned_graph = gated_edge_index.detach().cpu()
                batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
                gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
                gcn_outs.append(gcn_out)
                continue

            if self.graph_mode == 'prior_mask':
                gated_edge_index = self.get_prior_masked_edge_index(cos_ji_mat, edge_index, node_num, device)
                self.learned_graph = gated_edge_index.detach().cpu()
                batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
                gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
                gcn_outs.append(gcn_out)
                continue

            topk_num = min(self.topk, node_num)
            topk_num = self.topk#The value of k can be chosen by the user according to the desired sparsity level期望稀疏水平

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]#获取按照最后一个维度的前k个最大值以及索引

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)#得到多张图分批次的索引
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
            #gcn输出层使用gnn层
            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)


        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
   

        return out
        
