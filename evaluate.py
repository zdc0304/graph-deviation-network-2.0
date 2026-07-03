from util.data import *
import numpy as np
from util.simple_metrics import precision_score, recall_score, roc_auc_score, f1_score


# 4.3 evaluate
def get_full_err_scores(test_result, val_result):#all_scores表示所有节点在每个时间点的误差（预测值和真实值之间的误差）
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores = None
    all_normals = None
    feature_num = np_test_result.shape[-1]  # 节点数量

    labels = np_test_result[2, :, 0].tolist()

    for i in range(feature_num):
        test_re_list = np_test_result[:2, :, i]
        val_re_list = np_val_result[:2, :, i]

        scores = get_err_scores(test_re_list)  # 得到测试集的误差分数
        normal_dist = get_err_scores(val_re_list)  # 得到验证集的误差分数，

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))  # 沿竖直方向堆叠数组，增加行数
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    return all_scores, all_normals,labels


def get_final_err_scores(test_result, val_result):  # 最终的err分数
    full_scores, all_normals = get_full_err_scores(test_result, val_result)

    all_scores = np.max(full_scores, axis=0)

    return all_scores


def get_err_scores(test_res):  # Graph Deviation Scoring
    test_predict, test_gt = test_res
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)  # 得到err的中位数和方差

    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))  # 计算差值
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)  # robust normalization

    smoothed_err_scores = np.zeros(err_scores.shape)  # .shape返回array的行列等
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])  # 继续再小范围内取一个平均，加工err_scores

    return smoothed_err_scores


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)  # 均方误差损失


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[-topk:]

    topk_indices = np.transpose(topk_indices)  # 转置

    total_topk_err_scores = []
    topk_err_score_map = []
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
        sum_score = sum(
            score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])))

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)  # 计算f1分数

    return final_topk_fmeas


def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[
                   -topk:]  # 部分排序;取出前k个最大的元素的索引

    total_topk_err_scores = []
    topk_err_score_map = []

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)  # 每个时间步异常得分top k个的总和

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))  # 预测的标签
    pred_labels[total_topk_err_scores > thresold] = 1  # 大于阈值则设为1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])  # 取整

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)  # 越接近1，模型训练效果越好

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[
                   -topk:]  # 部分排序;取出前k个最大的元素的索引

    total_topk_err_scores = []
    topk_err_score_map = []

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0),
                                   axis=0)  # 每个时间步top k个err_scores求和

    final_topk_fmeas, thresolds = eval_scores(total_topk_err_scores, gt_labels, 400,
                                              return_thresold=True)  # 传入异常得分排名前k个的值，最后得到一个含有很多f1分数的列表 以及阈值列表

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))  # 获取f1得分最高的阈值步
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))  # 是否受到attack
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])  # 变成int型

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)  # 得到precise and recall的值

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)  # 得到auc分数

    return max(final_topk_fmeas), pre, rec, auc_score, thresold
