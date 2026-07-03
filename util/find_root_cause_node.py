
from util.data import get_attack_interval
import time
from datetime import datetime
from pytz import utc, timezone
from util.time import timestamp2str
def find_root_cause_node(fc_struc, total_err_scores, time_period_max_sensor,slide_win,down_len):
    def find_max_error_neighbor(sensor, time_head, time_end,fc_struc, total_err_scores, visited=None):
        if visited is None:
            visited = set()  # 用集合记录已访问的传感器 ，防止重复访问
        neighbors = fc_struc.get(sensor, [])#sensor不存在则返回一个空列表
        max_error_neighbor = None#
        max_error = float('-inf')
        for neighbor in neighbors:
            if neighbor not in fc_struc or neighbor in visited:  # 如果邻居传感器已经访问过，则跳过
                continue
            sum_error_score= sum(total_err_scores[time_head:time_end + 1][neighbor])
            if sum_error_score > max_error:
                max_error = sum_error_score
                max_error_neighbor = neighbor
        visited.add(sensor)  # 将当前传感器添加到已访问集合中
        return max_error_neighbor

    def analyze_attack(fc_struc, total_err_scores, time_period_max_sensor):
        result = []
        visited = set()
        for period_info in time_period_max_sensor:
            # 获取除去头部和尾部的键值对
            filtered_period_info = {k: v for k, v in period_info.items() if k not in ['head', 'end']}
            # 找出出现次数最多的传感器名字
            max_error_sensor = max(filtered_period_info.items(), key=lambda x: x[1])[0]
            head = period_info['head']
            end = period_info['end']
            head_t = timestamp2str(start_s + (head + slide_win) * down_len, fmt, cst8)
            end_t = timestamp2str(start_s + (end + slide_win) * down_len, fmt, cst8)
            time_head = head
            time_end = end
            max_error_neighbor = find_max_error_neighbor(max_error_sensor, time_head, time_end,fc_struc, total_err_scores, visited=visited)
            period_result = {}
            period_result['head_t'] = head_t
            period_result['end_t'] = end_t
            if max_error_neighbor is None:
                period_result[max_error_sensor] = max_error_neighbor#无邻居情况
            while max_error_neighbor is not None:
                visited.add(max_error_sensor)
                period_result[max_error_sensor] = max_error_neighbor
                max_error_sensor=max_error_neighbor
                max_error_neighbor = find_max_error_neighbor(max_error_sensor, time_head, time_end, fc_struc,
                                                             total_err_scores,visited=visited)
                if max_error_neighbor is not None:
                    period_result[max_error_sensor] = max_error_neighbor
            result.append(period_result)
        return result
    s = '2022/8/12 22:00:00'
    start_s = int(time.mktime(datetime.strptime(s, "%Y/%m/%d %H:%M:%S").timetuple()))
    cst8 = timezone('Asia/Shanghai')
    fmt = '%m/%d %H:%M:%S'
    result = analyze_attack(fc_struc, total_err_scores, time_period_max_sensor)
    print(result)
