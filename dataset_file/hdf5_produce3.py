import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 定义文件夹路径
data_folder_total = r"D:\Desktop\AMC competition\data_fil_total"  # 替换为你的总文件夹路径

# 定义 get_amp_phase 函数


# 处理单个 .csv 文件
def process_csv(file_path):
    # 读取 .csv 文件
    df = pd.read_csv(file_path, header=None)  # 假设 .csv 文件没有表头

    # 提取 IQ 信号（第一列和第二列），转换为 2*L 的 numpy 数组
    iq_signal = df.iloc[:, :2].values.T  # 转置为 2*L 形状


    # 提取码序列（第三列），去除空缺值并转换为 numpy 数组
    code_sequence = df.iloc[:, 2].dropna().values  # 去除空缺值并转换为数组

    # 提取调制方式编号（第四列）和码元宽度（第五列），只有第一行有值
    modulation_type = df.iloc[0, 3] if not pd.isna(df.iloc[0, 3]) else None
    symbol_width = df.iloc[0, 4] if not pd.isna(df.iloc[0, 4]) else None

    # 将数据组织为一个字典
    data_dict = {
        "iq_signal": iq_signal,
        "code_sequence": code_sequence,
        "modulation_type": modulation_type,
        "symbol_width": symbol_width,
    }

    return data_dict

# 递归遍历文件夹及其子文件夹，并按比例划分数据集
def process_folder(folder_path):
    data_list = []
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            # 如果是子文件夹，递归处理
            sub_data_list = process_folder(full_path)
            data_list.extend(sub_data_list)  # 使用 extend 合并数据
        elif entry.endswith(".csv"):
            # 如果是 .csv 文件，读取数据
            data_dict = process_csv(full_path)
            data_list.append(data_dict)
    return data_list

# 从每个子文件夹中以 6:2:2 的比例划分数据集
def split_data(data_list):
    train_data, test_valid_data = train_test_split(data_list, test_size=0.4, random_state=42)
    valid_data, test_data = train_test_split(test_valid_data, test_size=0.5, random_state=42)
    return train_data, valid_data, test_data

# 处理总文件夹中的所有子文件夹
all_data_list = process_folder(data_folder_total)

# 按 6:2:2 的比例划分数据集
train_data, valid_data, test_data = split_data(all_data_list)

# 将数据列表转换为 DataFrame
df_train = pd.DataFrame(train_data, columns=["iq_signal", "code_sequence", "modulation_type", "symbol_width"])
df_valid = pd.DataFrame(valid_data, columns=["iq_signal", "code_sequence", "modulation_type", "symbol_width"])
df_test = pd.DataFrame(test_data, columns=["iq_signal", "code_sequence", "modulation_type", "symbol_width"])

# 查看读取的数据
print("Train Data:")
print(df_train.head())
print("\nValidation Data:")
print(df_valid.head())
print("\nTest Data:")
print(df_test.head())

# 将三个数据集保存到同一个 HDF5 文件中，并赋予不同的 key
hdf5_path = r"D:\Desktop\AMC competition\data_fil_total\data_fil_split3.h5"  # 文件保存路径
with pd.HDFStore(hdf5_path, mode="w") as store:
    store.put("train", df_train)  # 保存训练集，key 为 "train"
    store.put("valid", df_valid)  # 保存验证集，key 为 "valid"
    store.put("test", df_test)  # 保存测试集，key 为 "test"

print(f"数据已保存到 {hdf5_path}")