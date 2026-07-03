import os
import subprocess

gpu_n = os.getenv('cuda', 'cpu')
dataset = os.getenv('DATASET', 'hai23.05_end')
seed = 0
BATCH_SIZE = 5
SLIDE_WIN = 5
dim = 64
out_layer_num = 1
SLIDE_STRIDE = 1
topk = 15
out_layer_inter_dim = 128
val_ratio = 0.1
decay = 0
path_pattern = dataset
COMMENT = dataset
EPOCH = 50
report = 'best'

# 确定要使用的 GPU 设备
if gpu_n != "cpu":
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_n

for i in range(6725, 6800):
    print(f'the seed is {i}:')
    log_file = f"G:/GDN-main-fuben/data/log_seed_{i}_outlayer1.txt"  # 定义日志文件名，并指定保存路径
    with open(log_file, "w") as f:  # 创建日志文件
        command = f"python main.py -dataset {dataset} -save_path_pattern {path_pattern} -slide_stride {SLIDE_STRIDE} -slide_win {SLIDE_WIN} -batch {BATCH_SIZE} -epoch {EPOCH} -comment {COMMENT} -random_seed {i} -decay {decay} -dim {dim} -out_layer_num {out_layer_num} -out_layer_inter_dim {out_layer_inter_dim} -val_ratio {val_ratio} -report {report} -topk {topk} -device {gpu_n}"

        if gpu_n == "cpu":
            subprocess.run(command, shell=True, stdout=f)
        else:
            subprocess.run(command, shell=True, env=env, stdout=f)
