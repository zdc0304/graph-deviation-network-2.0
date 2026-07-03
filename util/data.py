# util functions about data

from util.simple_metrics import f1_score, mean_squared_error, iqr, trim_mean
import numpy as np
from numpy import percentile


def rankdata(values, method='ordinal'):
    values = np.asarray(values)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def get_attack_interval(attack): #得到受到攻击的时间间隔
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i-1] == 0:
                heads.append(i)#第一次受到攻击的时间点
            
            if i < len(attack)-1 and attack[i+1] == 0:
                tails.append(i)#最后一次受到攻击的时间点
            elif i == len(attack)-1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))#确定攻击的起始位置和结束位置
    # print(heads, tails)
    return res

# calculate F1 scores
#  3.6 Graph Deviation Scoring
def eval_scores(scores, true_scores, th_steps, return_thresold=False):#得到一个含有很多f1分数的列表和阈值列表
    padding_list = [0]*(len(true_scores) - len(scores))#传入的scores是取了每个时间步top k个最大的error的总和
    # print(padding_list)

    if len(padding_list) > 0:#补齐
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method='ordinal')#返回各元素按从小到大排列的排名
    th_steps = th_steps#传入阈值步数,用于生成阈值列表
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps#生成一个（0，1）范围的浮点数数组，步长为1.0/th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)#>后边是不同的阈值

        fmeas[i] = f1_score(true_scores, cur_pred)#传入label（truth）和 预测（是否受到attack），计算f1分数，得到一个列表# F1 =(2×Prec×Rec)/(Prec+Rec)有自带的库函数计算

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))#找到scores_sorted中第一个大于阈值的error score的索引
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas#这里return_thresold设置为false，则返回fmeas

def eval_mseloss(predicted, ground_truth):#计算均方差损失

    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)

    
    # mask = (ground_truth_list == 0) | (predicted_list == 0)

    # ground_truth_list = ground_truth_list[~mask]
    # predicted_list = predicted_list[~mask]

    # neg_mask = predicted_list < 0
    # predicted_list[neg_mask] = 0

    # err = np.abs(predicted_list / ground_truth_list - 1)
    # acc = (1 - np.mean(err))

    # return loss
    loss = mean_squared_error(predicted_list, ground_truth_list)#均方误差计算loss 3.5 out layer部分


    return loss
#以下是3.6部分，Graph Deviation Scoring以及计算f1分数
def get_err_median_and_iqr(predicted, groundtruth):#error的中位数和方差

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr

def get_err_median_and_quantile(predicted, groundtruth, percentage):#得到error的的中位数和分位数

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))#percentage处的值和1-percentage处的值相减

    return err_median, err_delta

def get_err_mean_and_quantile(predicted, groundtruth, percentage):#得到err的均值和分位数

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)#先去除特定范围的两端数据，再取平均
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta

def get_err_mean_and_std(predicted, groundtruth):#得到error的均值和标准差

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):#contamination表示数据中异常值的比例

    padding_list = [0]*(len(gt) - len(scores))#需填充的列表
    # print(padding_list)

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype('int').ravel()#大于阈值则设为1（异常），小于阈值则设为0（正常）

    return f1_score(gt, pred_labels)
