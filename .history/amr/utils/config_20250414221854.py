from yacs.config import CfgNode as CN
import argparse
import importlib

__all__ = ['get_cfgs']

_C = CN()

_C.modes = CN()
_C.modes.method = 'DAELSTM'
_C.modes.path = ''
_C.modes.loss = 'loss_CE'
_C.modes.optimizer = 'adamw'
_C.modes.train = False
_C.modes.load_pretrained = False
_C.modes.load_pretrained_pos = False
_C.modes.load_model_width = False
_C.modes.training_pretrained_path = ''
_C.modes.testing_pretrained_path = ''
_C.modes.model_width_path = ''
_C.model.result_save_path = ''

_C.data_settings = CN()
_C.data_settings.dataset = 'ours'
_C.data_settings.Xmode = CN()
_C.data_settings.Xmode.type = 'IQ'
_C.data_settings.Xmode.options = CN()
_C.data_settings.Xmode.options.IQ_norm = False
_C.data_settings.Xmode.options.zero_mask = False
_C.data_settings.hdf5_path = ''
_C.data_settings.mod_type = ['BPSK', 'QPSK', '8PSK', 'MSK', '8QAM', '16QAM', '32QAM', '8APSK', '16APSK', '32APSK','others']

_C.opt_params = CN()
_C.opt_params.batch_size = 512
_C.opt_params.epochs = 150
_C.opt_params.lr = 1e-3
_C.opt_params.workers = 0
_C.opt_params.seed = 1
_C.opt_params.gpu = 0
_C.opt_params.cpu = False
_C.opt_params.noise_flag = False
_C.opt_params.early_stop = False
_C.opt_params.stop_patience = 50


def get_cfg_defaults():
    return _C.clone()

def get_cfgs():
    cfgs = get_cfg_defaults()
    parser = argparse.ArgumentParser(description='AMR HyperParameters')
    parser.add_argument('--config', type=str, default='configs/transformerlstm_ours.yaml',
                        help='type of config file. e.g. transformerlstm_16 (configs/transformerlstm_16.yaml)')
    opt, unparsed = parser.parse_known_args()
    with open(opt.config, "r", encoding="utf-8") as f:
        cfg_content = f.read()
        other_cfgs = CN.load_cfg(cfg_content)
    
    # 合并配置
    cfgs.merge_from_other_cfg(other_cfgs)
    return cfgs
