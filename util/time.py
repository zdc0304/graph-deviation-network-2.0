import time
import math
from datetime import datetime
from pytz import utc, timezone

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSincePlus(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timestamp2str(sec, fmt, tz):
    return datetime.fromtimestamp(sec).astimezone(tz).strftime(fmt)
# timestamp = 1618413316  # 2021-04-14 10:48:36 (UTC 时间)
# format_string = "%Y-%m-%d %H:%M:%S"
# timezone_info = cst8  # 中国标准时间（东八区）
#
# time_string = timestamp2str(timestamp, format_string, timezone_info)
# print(time_string)  ->2021-04-14 18:48:36
