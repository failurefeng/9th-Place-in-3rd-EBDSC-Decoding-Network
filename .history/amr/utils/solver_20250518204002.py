from amr.utils import logger
import torch
import time
from amr.utils.static import *
import os
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import csv
import math
import signal

from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
__all__ = ["Trainer", "Tester"]



def dynamic_split(self, signal_batch: torch.Tensor, signal_lens: list, symbol_widths: torch.Tensor, device,):
    """批处理的分割"""
    # signal_batch: (B, 2, L)
    expected_splits = []
    expected_nums = []
    expected_widths = []

    for i in range(signal_batch.size(0)):
        signal = signal_batch[i,:,:signal_lens[i]].clone() # (2 * L)
        symbol_width = symbol_widths[i].clone() if signal_batch.size(0) > 1 else symbol_widths.clone()

        symbol_width = symbol_width.clamp(min=0.25, max=1)
        width = int((symbol_width * 20).round())
        num_split = int((signal_lens[i] / width).ceil())
        expected_len = num_split * width
        
        if expected_len != signal_lens[i]:
            expected_split = F.pad(signal, (0, expected_len - signal_lens[i]), "constant", 0)
        else:
            expected_split = signal
        expected_splits.append(expected_split)
        expected_nums.append(num_split)
        expected_widths.append(width)

    max_num = max(expected_nums)
    max_width = max(expected_widths)
    max_width = max_width + (max_width % (2)) # 保证最大宽度是2的倍数

    splits = []
    for i, expected_split in enumerate(expected_splits):
        split = expected_split.view(2, expected_nums[i], expected_widths[i]).to(device)
        split = F.pad(split, (0, max_width - expected_widths[i], 0, max_num-expected_nums[i]), "constant", 0) # 码元宽度作为最后一个维度，它的填充长度要放在前面
        splits.append(split)
    split_tensor = torch.stack(splits, dim=0).to(device) # (B, 2, N, W)

    # 生成mask
    split_pad_mask = torch.ones(signal_batch.size(0), max_num, dtype=torch.bool)
    for i in range(signal_batch.size(0)):
        split_pad_mask[i, :expected_nums[i]] = False

    return split_tensor.to(self.device), split_pad_mask.to(self.device)


def correct_iq_phase(signal_batch: torch.Tensor, signal_lens: list, device):
    """基于前导训练序列的相位估计"""
    # padded_signals: (B, 2, L)
    padded_signals = signal_batch.clone()
    for i in range(padded_signals.size(0)):
        # 取前100个采样点估计相位
        sample_num = min(signal_lens[i], 100)
        pilot_samples = padded_signals[i, :, :sample_num]
        complex_pilot = pilot_samples[0] + 1j * pilot_samples[1]

        if torch.any((complex_pilot).abs() > 1e-6):
            # 使用最小二乘估计
            sum_p = torch.sum(complex_pilot * torch.conj(complex_pilot[0]), dim=0)
            phase = torch.angle(sum_p)
        else:
            phase = 0.0

        rotation_matrix = torch.tensor([
            [torch.cos(phase), torch.sin(phase)],
            [-torch.sin(phase), torch.cos(phase)]
        ])
        useful_signal = padded_signals[i, :, :signal_lens[i]]
        padded_signals[i, :, :signal_lens[i]] = rotation_matrix.to(device) @ useful_signal.to(device)
    # 标准化
    padded_signals /= 0.63
    return padded_signals.to(device)

class Trainer:
    def __init__(self, model, model_width, device, optimizer, base_lr, warmup_scheduler, main_scheduler, warmup_epochs, criterion, save_path, valid_freq=1, early_stop=True, stop_patience= 150, noise_flag=None, mask_flag=False, batch_size=512, loss_name = None):
        self.model = model
        self.model_width = model_width
        self.device = device
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.criterion = criterion
        self.all_epoch = None
        self.cur_epoch = 1
        self.train_loss = None
        self.train_CE = None
        self.train_CQ_score = None
        self.valid_loss = None
        self.valid_CE = None
        self.valid_CQ_score = None
        self.scheduler_val_loss = 10
        self.train_loss_all = []
        self.train_CE_all = []
        self.train_CQ_score_all = []
        self.valid_loss_all = []
        self.valid_CE_all = []
        self.valid_CQ_score_all = []
        self.save_path = save_path
        self.best_CE = 0.0
        self.best_CQ_score = None
        self.best_loss = None
        # early stopping
        self.early_stop = early_stop
        self.stop_patience = stop_patience #! ?个epoch没有提升就停止
        self.counter = 0
        self.stop_flag = False
        self.loss_name = loss_name
        self.batch_size = batch_size
        self.best_model_name = None # 保存最好的模型的名字 
        self.base_lr = base_lr
        
        os.makedirs(save_path, exist_ok=True)
        #-------------------------记录训练结果------------------------------
        self.result_file = os.path.join(self.save_path, "training_progress.csv")
        # 初始化时创建CSV文件并写入标题
        if not os.path.exists(self.result_file):
            with open(self.result_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train_CE", "Valid_CE", "Train_CQ_score", "Valid_CQ_score"])
        #--------------------------------------------------------------------------
        print("mask_flag = ", mask_flag)
        print("early_stop = ", early_stop)
        
        self.scaler = GradScaler()
        self.noise_flag = noise_flag
        self.mask_flag = mask_flag # 是否使用随机置0给信号加噪声，进行数据增强
        self.warmup_epochs = warmup_epochs
        self.reset_epochs = 15


    def loop(self, epochs, train_loader, valid_loader):
        self.all_epoch = epochs
        self.zero_mask_epoch = int(10) #！随机置0的时机
        self.rely_epoch = int(0)  #！引入码元宽度预测网络的时机

        for ep in range(self.cur_epoch, epochs+1):
            self.cur_epoch = ep
            if self.loss_name == "loss_CE":
                self.train_loss, self.train_CE, self.train_rounded_CE = self.train(train_loader)
                self.train_CE_all.append(self.train_CE)
                self.train_CQ_score_all.append(self.train_CQ_score)
            else:
                print("in function 'loop()', loss_name is not defined")
            self.train_loss_all.append(self.train_loss)
            

            if self.loss_name == "loss_CE":
                self.valid_loss, self.valid_CE, self.train_CQ_score = self.val(valid_loader)
                self.valid_CE_all.append(self.valid_CE)
                self.valid_CQ_score_all.append(self.valid_CQ_score)
            else:
                print("in function 'loop()', loss_name is not defined")
            self.valid_loss_all.append(self.valid_loss)
            self._loop_postprocessing(self.valid_CE)
            #-----------------------每次epoch结束后追加写入数据----------------
            with open(self.result_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.cur_epoch,
                    self.train_CE,  # 需要确保这些属性已更新
                    self.valid_CE,
                    self.train_CQ_score,
                    self.valid_CQ_score
                ])
                # 强制刷新写入磁盘
                f.flush()
                os.fsync(f.fileno())
            #----------------------------------------------------------------
            if self.early_stop and self.stop_flag:
                print(f'early stopping at Epoch: [{self.cur_epoch}]')
                break
        return self.train_loss_all, self.train_CE_all, self.train_CQ_score, self.valid_loss_all, self.valid_CE_all, self.valid_CQ_score_all



    def zero_mask(self, Padded_signals_train):
        if self.cur_epoch < 15:
            p = 0.05
        elif self.cur_epoch < 75:
            p = 0.09
        elif self.cur_epoch < 90:
            p = 0.12
        else:
            p = 0.05
        num = int(Padded_signals_train.shape[2] * p)
        res = Padded_signals_train.clone()
        index = np.array([[i for i in range(Padded_signals_train.shape[2])] for _ in range(Padded_signals_train.shape[0])])
        for i in range(index.shape[0]):
            np.random.shuffle(index[i, :])
        for i in range(res.shape[0]):
            res[i, :, index[i, :num]] = 0
        
        return res

    def train(self, train_loader):
        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def lr_reset(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * (0.9 ** (self.cur_epoch//self.reset_epochs))  # 每次递减20%

    def _iteration(self, data_loader):
        
        iter_loss = AverageMeter('Iter loss')
        iter_CE = AverageMeter('Iter CE')
        iter_CQ_score = AverageMeter('Iter CQ_score')
        iter_lr = AverageMeter('Iter lr')
        stime = time.time()
        for batch_idx, batch in enumerate(data_loader):
            t_satart = time.time()

            Padded_signals = batch['padded_signals']
            padded_C = batch['padded_code_sequences']
            code_Mask = batch['code_pad_mask']
            signal_lens = batch['signal_lens']
            symbol_widths = batch['symbol_widths']
            mixed_widths = symbol_widths.clone()

            alpha = 0.0
            std = 0.0
            if (self.model_width is not None) :

                if  (self.model.training ):
                    self.model_width.eval()
                    self.model_width.to(self.device)
                    Corrected_signals = correct_iq_phase(Padded_signals, signal_lens, self.device)
                    pred_widths = self.model_width(Corrected_signals, signal_lens).detach().clone()
                    torch.cuda.synchronize()  # 显式等待GPU计算完成
                    pred_widths = pred_widths.to(self.device)
                    

                    alpha = 1.0
                    mixed_widths = alpha * pred_widths + (1-alpha) * symbol_widths.clone()    
                    # alpha = min(0.1 + (1-0.96**self.cur_epoch), 1.0)

                else:
                    self.model_width.eval()
                    self.model_width.to(self.device)
                    Corrected_signals = correct_iq_phase(Padded_signals, signal_lens, self.device)
                    pred_widths = self.model_width(Corrected_signals, signal_lens).detach().clone()
                    torch.cuda.synchronize()  # 显式等待GPU计算完成
                    pred_widths = pred_widths.to(self.device)

                    alpha = 1.0
                    mixed_widths = alpha * pred_widths + (1-alpha) * symbol_widths.clone() 
            
            if self.noise_flag:#! 训练阶段添加噪声
                if  (self.model.training):
                    std = 0.5
                else:
                    std = 0.0
                
                width_noise = (torch.randn(1, device=self.device) * 0.05 +torch.abs(torch.randn(1, device=self.device)) * 0.05 * 0.1) * std                 
                mixed_widths = symbol_widths.clone() + width_noise

            
            S, split_Mask = dynamic_split(self, Padded_signals, signal_lens, mixed_widths, self.device)
            padded_C, code_Mask = padded_C.to(self.device), code_Mask.to(self.device)

            if self.mask_flag and self.model.training and (self.cur_epoch > self.zero_mask_epoch):
                Padded_signals = self.zero_mask(Padded_signals)
            
            # 预测码元编码序列

            if self.loss_name == 'loss_CE': #! 开始前向传播
                with autocast():
                    C_pred = self.model(S, split_Mask)
                    if self.model.training:
                        assert C_pred.requires_grad, "模型输出未保留梯度!"
                    C_pred = C_pred.to(self.device)
                    loss, CE, CQ_score = self.criterion(padded_C, C_pred, code_Mask)
            else:
                print(f"Invalid model_name value: {self.loss_name}")
                raise ValueError("Invalid model_name value")

            if self.model.training:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
               
            iter_loss.update(loss)
            iter_CE.update(CE)
            iter_CQ_score.update(CQ_score)
            iter_lr.update(self.optimizer.state_dict()['param_groups'][0]['lr'])
            t_end = time.time()
            if t_end-t_satart>4:
                print(f"batch:{batch_idx} training finishied, time:{t_end-t_satart:.4f}")
        
        if not self.model.training: # 验证阶段统一调度
            self.scheduler_val_loss = iter_loss.avg

            if self.cur_epoch % self.reset_epochs ==0 and self.cur_epoch >= self.reset_epochs: # 每隔reset_epochs个epoch重置一次学习率, 
                self.lr_reset()
            if self.cur_epoch <= self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.main_scheduler.step(self.scheduler_val_loss)
        ftime = time.time()

        if self.loss_name == "loss_CE":
            if self.model.training:
                print(f'Train | '
                            f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                            f'loss: {iter_loss.avg:.4e} | '
                            f'CE: {iter_CE.avg:.4f} | '
                            f'CQ_score: {iter_CQ_score.avg:.4f} | '
                            f'Lr: {iter_lr.avg:.4e} | '
                            f'alpha:{alpha} | '
                            f'noise_std:{std} | '
                            f'Counter: {self.counter} | '
                            f'time: {ftime-stime:.4f}')
            else:
                print(f'Valid | '
                            f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
                            f'loss: {iter_loss.avg:.4e} | '
                            f'CE: {iter_CE.avg:.4f} | '
                            f'CQ_score: {iter_CQ_score.avg:.4f} | '
                            f'noise_std:{std} | '
                            f'time: {ftime-stime:.4f}')
        else:
            print("in _iteration, Invalid model_name value")
            raise ValueError("Invalid model_name value")

        return iter_loss.avg.item(), iter_CE.avg.item(), iter_CQ_score.avg.item()

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        full_path = f"{self.save_path}/{name}"
        torch.save(state, full_path)
        print(f"best CE update to {self.best_CE:.4f},model saved to {full_path}.")
    def get_best_model_path(self):
        return os.path.join(self.save_path, self.best_model_name)

    def _loop_postprocessing(self, CE):
        self.last_valid_CE = CE
        state = {
            'time': time.time(),
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_CE': self.best_CE
        }
        if self.best_CE is None and self.cur_epoch > self.rely_epoch:
            self.best_CE = CE
            state['best_CE'] = self.best_CE
            self.best_model_name=f"best_CQ.pth"
            self._save(state, name=self.best_model_name)
        elif CE >= self.best_CE: # CE不下降
            self.counter += 1
            if self.counter > self.stop_patience:
                self.stop_flag = True

        else:
            self.best_CE = CE
            state['best_CE'] = self.best_CE
            self.best_model_name=f"best_CE.pth"
            self._save(state, name=self.best_model_name)
            self.counter = 0

        
        



class Tester:
    def __init__(self, model, model_width, device, criterion, batch_size=512, loss_name = None, is_debug_sample = True, debug_sample_num = 20):
        self.model = model
        self.model_width = model_width
        self.device = device
        self.criterion = criterion
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.is_debug_sample = is_debug_sample  # 是否保存调试样本
        self.debug_sample_num = debug_sample_num      # 最多保存样本数
        self.debug_data = []                          # 存储调试数据

    def __call__(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            loss, CE, CQ_score = self._iteration(test_loader)

        return loss, CE, CQ_score

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_CE = AverageMeter('Iter CE')
        iter_CQ_score = AverageMeter('Iter CQ_score')
        iter_rely_loss = AverageMeter('Iter rely_loss')
        iter_rely_CE = AverageMeter('Iter rely_CE')
        iter_rely_CQ_score = AverageMeter('Iter rely_CQ_score')
        stime = time.time()
        for batch_idx, batch in enumerate(data_loader):

            Padded_signals = batch['padded_signals']
            padded_C = batch['padded_code_sequences']
            code_Mask = batch['code_pad_mask']
            signal_lens = batch['signal_lens']
            symbol_widths = batch['symbol_widths']
            padded_C, code_Mask = padded_C.to(self.device), code_Mask.to(self.device)
            # 在码元宽度预测网络的预测结果的基础上进行测试-----------------------------------------------------------------------
            if self.model_width is not None:
                self.model_width.eval()
                self.model_width.to(self.device)
                Corrected_signals = correct_iq_phase(Padded_signals, signal_lens, self.device)
                pred_widths = self.model_width(Corrected_signals, signal_lens).detach().clone()
                torch.cuda.synchronize()  # 显式等待GPU计算完成
                pred_widths = pred_widths.to(self.device)
                rely_S, rely_split_Mask = dynamic_split(self, Padded_signals, signal_lens, pred_widths, self.device)
                # 预测码元编码序列      
                if self.loss_name == 'loss_CE':
                    with autocast():
                        rely_C_pred = self.model(rely_S, rely_split_Mask)
                        rely_C_pred =  rely_C_pred.to(self.device)
                        rely_loss, rely_CE, rely_CQ_score = self.criterion(padded_C, rely_C_pred, code_Mask)
                else:
                    print(f"Invalid model_name value: {self.loss_name}")
                    raise ValueError("Invalid loss_name value")
                iter_rely_loss.update(rely_loss)
                iter_rely_CE.update(rely_CE)
                iter_rely_CQ_score.update(rely_CQ_score)
            #-----------------------------------------------------------------------------------------------------------------
            # 将batch中的每个信号分割成码元切片并堆叠成一个张量，形状为 (num_symbol, 2, symbol_width)   
            S, split_Mask = dynamic_split(self, Padded_signals, signal_lens, symbol_widths, self.device)     
            
            # 预测码元编码序列      
            if self.loss_name == 'loss_CE':
                with autocast():
                    C_pred = self.model(S, split_Mask)
                    C_pred =  C_pred.to(self.device)
                    loss, CE, CQ_score = self.criterion(padded_C, C_pred, code_Mask)
            else:
                print(f"Invalid model_name value: {self.loss_name}")
                raise ValueError("Invalid loss_name value")
            
            iter_loss.update(loss)
            iter_CE.update(CE)
            iter_CQ_score.update(CQ_score)
        
    
        ftime = time.time()
        if self.loss_name == "CE":
            print(f'Test | '
                        f'loss: {iter_loss.avg:.4e} | '
                        f'CE: {iter_CE.avg:.4f} | '
                        f'CQ_score: {iter_CQ_score.avg:.4f} | '
                        f'time: {ftime-stime:.4f}')
            if self.model_width is not None:
                print('Test | '
                        f'rely_loss: {iter_rely_loss.avg:.4e} | '
                        f'rely_CE: {iter_rely_CE.avg:.4f} | '
                        f'rely_CQ_score: {iter_rely_CQ_score.avg:.4f} | ')
        else:
            print("in _iteration, Invalid model_name value")
        return iter_loss.avg.item(), iter_CE.avg.item(), iter_CQ_score.avg.item()



