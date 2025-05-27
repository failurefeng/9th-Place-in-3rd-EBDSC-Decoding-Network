import matplotlib.pyplot as plt
import matplotlib
import os
import torch
from amr.utils import logger
matplotlib.use('Agg')

def draw_train(train_loss, train_CE, valid_loss, valid_CE, save_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, "r.-", label="Training loss")
    plt.plot(valid_loss, "b.-", label="Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_CE, "r.-", label="Training CE")
    plt.plot(valid_CE, "b.-", label="Validation CE")
    plt.xlabel("epoch")
    plt.ylabel("CE")
    plt.legend()


    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'train_process.jpg'))
    logger.info(f'save the draw of training.')
    plt.close()
