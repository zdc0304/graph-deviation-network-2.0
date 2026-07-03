import os
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
train_visual = pd.read_csv('./HAI_train_VMD.csv', index_col=0)

# 获取数据集中的列名，即传感器的编号
sensors = train_visual.columns
# 每组的传感器数量
group_size = 10

# 创建保存图片的文件夹的根路径
output_folder_root = r'G:\GDN-main-fuben\data\visualization\hai_train_VMD'

# 遍历传感器，分组保存波形数据
for i in range(0, len(sensors), group_size):
    group_sensors = sensors[i:i + group_size]  # 获取当前组的传感器列表
    output_folder_name = f"{i + 1}-{i + len(group_sensors)}"  # 根据组号命名文件夹
    output_folder_path = os.path.join(output_folder_root, output_folder_name)  # 组合文件夹路径
    os.makedirs(output_folder_path, exist_ok=True)  # 创建文件夹，如果不存在则创建

    # 遍历当前组的传感器，绘制并保存波形图
    for sensor in group_sensors:
        try:
            plt.figure(figsize=(16, 2))  # 设置图表的大小
            plt.plot(train_visual.index, train_visual[sensor])  # 绘制折线图  [:2000]
            plt.title(f"Sensor {sensor} Data")  # 设置标题
            plt.xlabel("Time")  # 设置横坐标标签
            plt.ylabel("Value")  # 设置纵坐标标签
            plt.xticks(train_visual.index[::1000], rotation=45)  # 设置 x 轴刻度为每隔 1000 个数据点一个，并旋转刻度标签
            # 保存图片到文件夹
            output_file_path = os.path.join(output_folder_path, f"{sensor}.png")
            plt.savefig(output_file_path, bbox_inches='tight')

            plt.close()  # 关闭当前图表，释放资源
        except Exception as e:
            print(f"Error plotting data for sensor {sensor}: {str(e)}")




