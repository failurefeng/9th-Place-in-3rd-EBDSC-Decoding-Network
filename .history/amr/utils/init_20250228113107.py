import torch
import numpy as np
import random
import os
from amr.utils import logger
from amr.models import *
import importlib
from amr.models.networks.width_net_old1 import HybridModel_width

__all__ = ["init_device", "init_model","init_model_width", "init_loss"]


def init_device(seed=None, cpu=None, gpu=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):

    if args.modes.train:
        model = getattr(
                importlib.import_module("amr.models.networks." + args.modes.method + '_' + args.data_settings.dataset),
                args.modes.method)() # 
        state_dict = model.state_dict() 
        if args.modes.load_pretrained:
            pretrained = args.modes.training_load_pretrained_path
            assert os.path.isfile(pretrained)
            pre_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
            # 只加载和当前模型结构一致的参数
            for name, param in pre_state_dict.items():
                if name in state_dict and state_dict[name].shape == param.shape:
                    state_dict[name] =param           
            model.load_state_dict(state_dict,strict=False)
            logger.info("pretrained model loaded from {}".format(pretrained))
        if args.modes.load_pretrained_pos:
            # 加载预训练模型pos_embedding的部分参数，并线性插值到1*256*n_model的形状
            pretrained = './results/TransformerLSTM/best/RML2018/checkpoints/best_acc.pth'
            assert os.path.isfile(pretrained)
            pre_pos_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
            pre_pos_embedding = pre_pos_state_dict['pos_embedding']
            pre_pos_embedding = pre_pos_embedding.permute(0, 2, 1)  # [1, n_model, 64]
            new_length = state_dict['pos_embedding'].shape[1]
            interpolated = torch.nn.functional.interpolate(
                pre_pos_embedding,
                size=new_length,
                mode='linear',  # 线性插值
                align_corners=False  # 是否对齐角点
            )
            # 恢复原始形状 [1, new_length, n_model]
            interpolated = interpolated.permute(0, 2, 1)
            state_dict['pos_embedding'] = interpolated 
            model.load_state_dict(state_dict,strict=False)
            logger.info("pretrained pos_embedding loaded from {}".format(pretrained))
        # print(model)
    if not args.modes.train:
        model = getattr(
                importlib.import_module("amr.models.networks." + args.modes.method + '_' + args.data_settings.dataset),
                args.modes.method)() # 
        pretrained = args.modes.testing_load_pretrained_path
        print("try to load model from:",pretrained)
        assert os.path.isfile(pretrained)
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(pretrained))

    return model

def init_model_width(args):
    if args.modes.load_model_width:
        model_width = HybridModel_width()
        pretrained = args.modes.load_model_width_path
        assert os.path.isfile(pretrained)
        pre_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['model_state']
        model_width.load_state_dict(pre_state_dict,strict=False)
        logger.info("pretrained width_net loaded from {}".format(pretrained))
    else:
        model_width = None
    return model_width

def init_loss(loss_func):
    loss = getattr(importlib.import_module("amr.models.losses.loss"), loss_func)()
    return loss


if __name__ == '__main__':
    model = getattr(importlib.import_module("amr.models.networks." + 'ResNet_RML2016'), "ResNet")(11)
    print(model)
