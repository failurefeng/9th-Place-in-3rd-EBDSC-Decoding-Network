import matplotlib.pyplot as plt
import matplotlib
import os
import torch
from amr.utils import logger
matplotlib.use('Agg')

__all__ = ["draw_train", "draw_conf", "draw_acc"]


def draw_train(train_loss, train_CQ_score, train_rounded_CQ_score, valid_loss, valid_CQ_score, valid_rounded_CQ_score, save_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, "r.-", label="Training loss")
    plt.plot(valid_loss, "b.-", label="Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_CQ_score, "r.-", label="Training CQ score")
    plt.plot(valid_CQ_score, "b.-", label="Validation CQ score")
    plt.xlabel("epoch")
    plt.ylabel("CQ score")
    plt.legend()


    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'train_process.jpg'))
    logger.info(f'save the draw of training.')
    plt.close()


def draw_conf(test_conf, save_path, cmap=plt.cm.Blues, labels=[], order=None):
    plt.figure(figsize=(16, 16))
    plt.imshow(test_conf, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = torch.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=10)
    plt.yticks(tick_marks, labels, fontsize=10)
    test_conf = test_conf.numpy().round(2)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if test_conf[i,j] == 0:
                continue
            if i == j:
                plt.text(j, i, test_conf[i, j], va='center', ha='center', color='white')  # 显示百分比
            else:
                plt.text(j, i, test_conf[i, j], va='center', ha='center')  # 显示百分比

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'test_conf_'+order+'.jpg'))
    logger.info(f'save the draw of confusion matrix.')
    plt.close()


def draw_acc(snrs, test_acc_snr, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(snrs, test_acc_snr, "bs-", label="Test acc")
    plt.xlabel("snr(dB)")
    plt.ylabel("acc")
    plt.legend()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'test_acc.jpg'))
    logger.info(f'save the draw of testing accuracy.')
    plt.close()
