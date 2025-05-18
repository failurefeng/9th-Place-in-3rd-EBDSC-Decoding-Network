import torch
import torch.nn as nn
from amr.dataloaders.dataloader2 import *
from amr.utils import *
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from complexity import *
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ReduceLROnPlateau, CosineAnnealingLR
from functools import wraps



def main(cfgs):
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization

    device, pin_memory = init_device(cfgs.opt_params.seed, cfgs.opt_params.cpu, cfgs.opt_params.gpu)
    print(device, pin_memory)

    train_loader, valid_loader, test_loader = AMRDataLoader(dataset=cfgs.data_settings.dataset,
                                                                        Xmode=cfgs.data_settings.Xmode,
                                                                        hdf5_path = r"D:\Desktop\AMC competition\data_fil_total\data_fil_split3.h5",
                                                                        batch_size=cfgs.opt_params.batch_size,
                                                                        num_workers=cfgs.opt_params.workers,
                                                                        pin_memory=pin_memory,
                                                                        ddp = cfgs.modes.ddp,
                                                                        random_mix = cfgs.data_settings.Xmode.options.random_mix,
                                                                        mod_type=cfgs.data_settings.mod_type,
                                                                        )()

    model = init_model(cfgs)
    model.to(device)

    #! 可以加载码元宽度预测网络
    model_width = init_model_width(cfgs) # 若不加载，返回None




    criterion = init_loss(cfgs.modes.loss)

    if cfgs.modes.train:
        print("training start...")

        if cfgs.modes.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs.opt_params.lr, weight_decay=1e-4, amsgrad=True)
        elif cfgs.modes.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfgs.opt_params.lr, momentum=0.8, weight_decay=0.02)
        else:
            print("no available optimizer")

        # 前?个epoch逐步预热（线性递增）
        warmup_epochs = int(1)

        scheduler1 = LinearLR(
            optimizer, 
            start_factor=0.8,  # 初始学习率为设置值的比例
            end_factor=1.0,     # 逐渐升温到完整学习率
            total_iters=warmup_epochs
        )

        # ReduceLROnPlateau主调度阶段
        scheduler2 = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.0001, threshold_mode='rel', min_lr=1e-7)
        scheduler3 = CosineAnnealingLR(optimizer, T_max=cfgs.opt_params.epochs, eta_min=1e-6)

        trainer = Trainer(model=model, model_width=model_width, device=device, optimizer=optimizer, warmup_scheduler=scheduler1, main_scheduler=scheduler2, warmup_epochs=warmup_epochs, criterion=criterion,
                          save_path='results/' + cfgs.modes.method + '/' + cfgs.modes.path + '/' + cfgs.data_settings.dataset + '/checkpoints',
                          early_stop=cfgs.opt_params.early_stop,
                          stop_patience = 50,
                          mask_flag = cfgs.data_settings.Xmode.options.zero_mask,
                          batch_size = cfgs.opt_params.batch_size,
                          loss_name = cfgs.modes.loss)
        train_loss, train_CQ_score, train_rounded_CQ_score, valid_loss, valid_CQ_score, valid_rounded_CQ_score = trainer.loop(cfgs.opt_params.epochs, train_loader, valid_loader)
        draw_train(train_loss, train_CQ_score, train_rounded_CQ_score, valid_loss, valid_CQ_score,valid_rounded_CQ_score,
                   save_path='./results/' + cfgs.modes.method + '/' + cfgs.modes.path + '/' + cfgs.data_settings.dataset + '/draws')


    # 测试前重新加载最优模型
    cfgs.modes.train = False

    model = init_model(cfgs)
    model.to(device)


    print("testing start...")
    test_loss, test_CQ_score, test_rounded_CQ_score = Tester(model=model, model_width=model_width, device=device, criterion=criterion,
                                                                         batch_size = cfgs.opt_params.batch_size,
                                                                         loss_name = cfgs.modes.loss)(test_loader)

    #! Y_pred 的形状是 (batch_size,)，每个值是一个整数，范围是 [0, num_classes - 1]，记得提交时全体要加上1

if __name__ == '__main__':
    cfgs = get_cfgs()
    main(cfgs)