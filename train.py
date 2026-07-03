import numpy as np
import torch
import torch.nn as nn
import time
from util.time import *
from util.env import *
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores,get_final_err_scores
from torch.utils.data import DataLoader, random_split, Subset
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def loss_func(y_pred, y_true):#调用均方差损失函数
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])#梯度优化器

    now = time.time()
    
    train_loss_list = []#损失
    cmp_loss_list = []

    device = get_device()#使用cpu or gpu 训练


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']#使用config 传入epoch
    early_stop_win = 2
    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader
    # 初始化存储每个epoch的损失和累积损失的列表
    avg_loss_per_epoch = []  # 存储每个epoch的平均损失
    acu_loss_per_epoch = []  # 存储每个epoch的累积损失
    for i_epoch in range(epoch):#开始训练
        acu_loss = 0
        model.train()

        for x, y, attack_label, edge_index in dataloader:
            _start = time.time()

            x, y, edge_index = [item.float().to(device) for item in [x, y, edge_index]]

            optimizer.zero_grad()#清零上一个epoch的梯度
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, y)
#           final_err_scores=get_final_err_scores(out , y)
            loss.backward()#反向传播计算梯度
            optimizer.step()#参数更新（校正）
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1

        # 计算每个epoch的平均损失
        avg_loss = acu_loss / len(dataloader)

        # 记录平均损失和累积损失
        avg_loss_per_epoch.append(avg_loss)
        acu_loss_per_epoch.append(acu_loss)
        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )
        #loss accuracy_loss visualization



        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)
            #验证效果更好则采用验证集的损失，否则改善次数+1
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss
    if plt is None:
        return train_loss_list

    # 在训练结束后进行可视化
    plt.figure(figsize=(10, 6))

    # 绘制平均损失图表
    plt.subplot(2, 1, 1)
    plt.plot(range(len(avg_loss_per_epoch)), avg_loss_per_epoch, label='Average Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Visualization')
    plt.legend()

    # 绘制累积损失图表
    plt.subplot(2, 1, 2)
    plt.plot(range(len(acu_loss_per_epoch)), acu_loss_per_epoch, label='Accumulated Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accumulated Loss')
    plt.title('Accumulated Loss Visualization')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}.loss.png')
    plt.close()


    return train_loss_list
