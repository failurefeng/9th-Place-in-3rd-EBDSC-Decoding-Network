import torch.utils.data as data
import torch
import pickle
import numpy as np
import pandas as pd
import h5py
import os
from torch.utils.data import DataLoader
from .transform import *
import random
import torch.nn.functional as F
from ..utils import *
from torch.utils.data import DataLoader, Sampler

__all__ = ['AMRDataLoader']

class LengthSortSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        # 生成排序后的索引列表
        self.sorted_indices = sorted(
            range(len(data_source)),
            key=lambda i: data_source[i]["signal"].shape[1]
        )
    
    def __iter__(self):
        # 生成排序后的批次（每个批次含多个索引）
        batches = [
            self.sorted_indices[i:i+self.batch_size] 
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        # 打乱批次顺序（保留批内排序，维持填充效率）
        random.shuffle(batches)
        yield from batches  # 直接返回索引列表的列表
    
    def __len__(self):
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


class PreFetcher:
    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            if isinstance(self.next_input, dict):
                for key, tensor in self.next_input.items():
                    if isinstance(tensor, torch.Tensor):
                        self.next_input[key] = tensor.cuda(non_blocking=True)
            elif isinstance(self.next_input, (list, tuple)):
                for idx, tensor in enumerate(self.next_input):
                    if isinstance(tensor, torch.Tensor):
                        self.next_input[idx] = tensor.cuda(non_blocking=True)
            else:
                raise TypeError(f"Unsupported input type: {type(self.next_input)}")

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration

        # 只对张量调用 record_stream
        if isinstance(input, dict):
            for key, tensor in input.items():
                if isinstance(tensor, torch.Tensor):
                    tensor.record_stream(torch.cuda.current_stream())
        elif isinstance(input, (list, tuple)):
            for tensor in input:
                if isinstance(tensor, torch.Tensor):
                    tensor.record_stream(torch.cuda.current_stream())
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

        self.preload()
        return input

class myDataset(data.Dataset):
    def __init__(self, dataframe, Xmode_type):
        self.dataframe = dataframe
        self.Xmode_type = Xmode_type
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # 提取数据
        signal_name = self.Xmode_type+"_signal"
        signal = torch.tensor(row[signal_name], dtype=torch.float32)  # 转换为 tensor
        code_sequence = torch.tensor(row["code_sequence"], dtype=torch.float32)  # 转换为 tensor
        modulation_type = torch.tensor(row["modulation_type"], dtype=torch.long)  # 转换为 tensor
        symbol_width = torch.tensor(row["symbol_width"], dtype=torch.float32)  # 转换为 tensor
        
        # 返回一个样本
        return {
            "signal": signal,
            "code_sequence": code_sequence,
            "modulation_type": modulation_type,
            "symbol_width": symbol_width,
        }

# 自定义 collate_fn

def dynamic_padding_collate_fn(batch):
    signals = [item["signal"] for item in batch]
    code_sequences = [item["code_sequence"] for item in batch]
    modulation_types = torch.stack([item["modulation_type"] for item in batch])
    symbol_widths = torch.stack([item["symbol_width"] for item in batch])
    signal_lens = torch.tensor([len(item["signal"][0]) for item in batch])
    code_lens = torch.tensor([len(item["code_sequence"]) for item in batch])

    # 找到该批次中 signal 的最大长度
    max_length = max(signal_lens)
    # 将最大长度调整为 2^4 的整数倍
    if max_length % (2**4) == 0:
        max_length = max_length 
    else:
        max_length = max_length + (2**4 - max_length % (2**4)) 

    max_code_length = max(code_lens)
 
    padded_signals = []
    padded_codes = []
    # padded_splits = []
    for i, item in enumerate(batch):
        signal = item["signal"]
        signal_pad_length = max_length - signal.shape[1]
        padded_signal = F.pad(signal, (0, signal_pad_length), "constant", 0) if signal_pad_length > 0 else signal
        padded_signals.append(padded_signal)

        code_pad_length = max_code_length - code_lens[i]
        pad_code = F.pad(code_sequences[i], (0, code_pad_length), "constant", 0) if code_pad_length > 0 else code_sequences[i]
        padded_codes.append(pad_code)

    code_pad_mask = torch.zeros(len(batch),max_code_length)
    for i in range(len(batch)):
        code_pad_mask[i, :code_lens[i]] = 1.0

    # 堆叠其他字段
    padded_signals = torch.stack(padded_signals)
    padded_codes = torch.stack(padded_codes)
    # padded_splits = torch.stack(padded_splits)

    
    # 返回一个字典
    return {
        "signals": signals,
        "padded_signals": padded_signals,
        "code_sequences": code_sequences,
        "modulation_types": modulation_types,
        "symbol_widths": symbol_widths,
        'signal_lens': signal_lens,
        'padded_code_sequences': padded_codes,
        'code_pad_mask': code_pad_mask,
    }


"""
重构思想：
1. AMRDataLoader用于从文件中读取数据并分为训练集：测试集：验证集=2:1:1。 可以通过mod_type选取需要的调试方式，可以全选，也可以挑选子集。
涉及到星座图生成可能比较慢，在这里还要选择是信息流还是星座图，如果是星座图的话，检查是否生成过星座图（方便下次载入），否则执行生成星座图的过程。
RML2016和RML2018的读取可以分开为两个，也可以合并为一个AMRDataLoader，合并为一个DataLoader用参数控制。
2. 读取到的数据在载入AMRDataset的时候，通过载入模式选择不同的操作，包括选取的数据是IQ还是AP，是否归一化，是否zero_mask等等定制化操作。
Xmode json结构定制:

3. 将返回的AMRDataset格式载入DataLoader形成不同的训练、测试、验证集
"""



class AMRDataLoader(object):
    """
    dataset: {RML2016,RML2018}
    Xmode: 定制化数据载入风格
        dict{
            type:{'IQ','AP','IQ_and_AP','star'} 载入数据类型；IQ->AP使用原始IQ信号
            options:{
                IQ_norm: bool IQ数据是否归一化到[0,1]
                zero_mask: bool 是否进行掩码操作以进行数据增强
            }
            options具有可扩展性
        }
    batch_size:
    num_workers:
    pin_memory:
    mod_type:挑选调试方式子集，若全选则为全集
    """
    def __init__(self, dataset, batch_size, num_workers, pin_memory, mod_type, Xmode, hdf5_path):
        print("dataset:",dataset)
        df_train = pd.read_hdf(hdf5_path, key="train")
        df_valid = pd.read_hdf(hdf5_path, key="valid")
        df_test = pd.read_hdf(hdf5_path, key="test")
        #查看读取的数据
        print(f"size of train_dataset: {len(df_train)}")
        print(f"size of valid_dataset: {len(df_valid)}")
        print(f"size of test_dataset: {len(df_test)}")
        self.Xmode_type = Xmode.type
        self.mods = mod_type

        train_dataset = myDataset(df_train,self.Xmode_type)
        valid_dataset = myDataset(df_valid,self.Xmode_type)
        test_dataset = myDataset(df_test,self.Xmode_type)


        self.train_loader = DataLoader(train_dataset,  num_workers=num_workers,batch_size=batch_size,
                                     pin_memory=pin_memory,  shuffle=True, collate_fn=dynamic_padding_collate_fn)# 使用动态填充的 collate_fn, 同时将batch中最大长度调整为 2^4 的整数倍
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=pin_memory, shuffle=False, collate_fn=dynamic_padding_collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                    pin_memory=pin_memory, shuffle=False, collate_fn=dynamic_padding_collate_fn)# 使用动态填充的 collate_fn, 同时将batch中最大长度调整为 2^4 的整数倍

        if pin_memory is True:
            self.train_loader = PreFetcher(self.train_loader)
            self.valid_loader = PreFetcher(self.valid_loader)
            self.test_loader = PreFetcher(self.test_loader)

    def __call__(self):
        return self.train_loader, self.valid_loader, self.test_loader


if __name__ == '__main__':
    cfgs = get_cfgs()
    train_loader, valild_loader, test_loader, mods = AMRDataLoader(dataset="ours",batch_size=100,num_workers=4, mod_type=cfgs.data_settings.mod_type,Xmode=cfgs.data_settings.Xmode, pin_memory=False)() # 最后一个 () 的作用是调用 AMRDataLoader 实例的 __call__ 方法
    print(train_loader)
    print(valild_loader)
    print(test_loader)
    print(mods)
    for batch in train_loader:
        print(batch["signal"].shape)  # 打印每个批次的 signal 形状
        print(batch["code_sequence"][0])
        print(batch['padded_code_sequences'].shape)
        print(batch['padded_splits'].shape)
        break  # 只检查第一个批次