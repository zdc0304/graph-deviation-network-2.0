from util.data import get_attack_interval
import time
from datetime import datetime
from pytz import utc, timezone
from util.time import timestamp2str
import json
import argparse
import numpy as np

def printsep():
    print('='*40+'\n')

def save_attack_infos(f1_scores, total_err_scores, labels, names, save_path, dataset, slide_win,down_len):


    s = '2022/8/12 22:00:00'
    # if dataset == 'wadi' or dataset == 'wadi2': 我使用的是hai数据集
    #     s = '09/10/2017 18:00:00'
    # elif dataset == 'swat':
    #     s = '28/12/2015 10:00:00'
    start_s = int(time.mktime(datetime.strptime(s, "%Y/%m/%d %H:%M:%S").timetuple()))
    cst8 = timezone('Asia/Shanghai')
    fmt = '%m/%d %H:%M:%S'

    attack_inters = get_attack_interval(labels)

    save_infos = {
        'total_best_f1_score': f1_scores[0],
        'total_best_f1_score_topk': f1_scores[1],
        'total_best_f1_score_all': f1_scores[2],
        'attacks': []
    }

    indices_map = names
    #一个时间步=10s
    indices = np.argmax(total_err_scores, axis=0).tolist()#total_err_scores shape为（255，28000），计算得到每一个时间步异常得分最高的传感器的名字 以名字作为索引
    anomaly_sensors = [ indices_map[index] for index in indices ]#取出有异常的传感器的名字

    topk = 3#改为3
    topk_indices = np.argpartition(total_err_scores, -topk, axis=0)[-topk:]#得到每个时间步异常得分最高的前k个传感器的索引
    topk_indices = np.transpose(topk_indices)#转置 使每一行代表一个时间步 每一列代表top k 个异常得分最高的传感器的名字

    topk_anomaly_sensors = []
    topk_err_score_map=[]
    for i, indexs in enumerate(topk_indices):#构建topk sensors集合
        # print(indexs) index表示当前时间步异常得分最高的前k个传感器的名字
        topk_anomaly_sensors.append([indices_map[index] for index in indexs])

        item = {}
        for sensor, index in zip(topk_anomaly_sensors[i],indexs):
            if sensor not in item:
                item[sensor] = total_err_scores[index, i]#传感器：error  当前时间步的对应异常得分top k 个传感器的异常分数

        topk_err_score_map.append(item)#得到 [传感器:error]  字典

    time_period_max_sensor = []#[{max_error_sensor:次数   head:  end:}  {}]
    for head, end in attack_inters:#得到攻击的起始位置
        attack_infos = {}
        topk_attack_infos = {}

        head_t = timestamp2str(start_s+(head+slide_win)*down_len, fmt, cst8)
        end_t = timestamp2str(start_s+(end+slide_win)*down_len, fmt, cst8)
        # head_t = datetime.fromtimestamp(start_s+head).astimezone(cst8).strftime(fmt)
        # end_t = datetime.fromtimestamp(start_s+end).astimezone(cst8).strftime(fmt)

        # print(f'\nattack from {head_t} to {end_t}:')
        
        for i in range(head, end):
            # t = datetime.fromtimestamp(start_s+i).astimezone(cst8).strftime(fmt)
            t = timestamp2str(start_s+(i+slide_win)*down_len, fmt, cst8)
            max_sensor = anomaly_sensors[i]
            topk_sensors = topk_anomaly_sensors[i]#一个带有当前时间步异常得分最高的前k个传感器的名字的列表

            if max_sensor not in attack_infos:
                attack_infos[max_sensor] = 0
            attack_infos[max_sensor] += 1

            # for anomaly_sensor in topk_sensors:
            #     if anomaly_sensor not in topk_attack_infos:
            #         topk_attack_infos[anomaly_sensor] = 0
            #     topk_attack_infos[anomaly_sensor] += 1

            for anomaly_sensor in topk_sensors:
                if anomaly_sensor not in topk_attack_infos:
                    topk_attack_infos[anomaly_sensor] = 0
                topk_attack_infos[anomaly_sensor] += topk_err_score_map[i][anomaly_sensor]#对应时间步对应传感器的异常分数
            # 将当前时间段的异常传感器统计添加到字典中

        # print('-------------------------------')
        # print(f'total top 5 attack sensors from {head_t} to {end_t}:')#
        sorted_attack_infos = {k: v for k, v in sorted(attack_infos.items(), reverse=True, key=lambda item: item[1])} #{max_error_sensor:次数  }
        sorted_topk_attack_infos = {k: v for k, v in sorted(topk_attack_infos.items(), reverse=True, key=lambda item: item[1])}
        # for key, count in sorted_attack_infos.items()[:5]:
        #     print(key, count)
           # {max_error_sensor:次数 head: end: }
        completed_sorted_attack_infos=sorted_attack_infos
        completed_sorted_attack_infos['head'] = head
        completed_sorted_attack_infos['end'] = end
        time_period_max_sensor.append(completed_sorted_attack_infos)
        save_infos['attacks'].append({
            'start': head_t,
            'end': end_t,
            'sensors': list(sorted_attack_infos),
            'topk_sensors': list(sorted_topk_attack_infos),
            'topk_scores': list(sorted_topk_attack_infos.values())
        })

    with open(save_path, 'w+') as outfile:
        json.dump(save_infos, outfile, indent=4)  
    return time_period_max_sensor