import pandas as pd
import torch
import os
import wandb
from model import *
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

# ============ wandb 配置 ============
WANDB_ENTITY = "r-wang-34-delft-university-of-technology"
WANDB_PROJECT = "sts2024-tooth-seg"


def trainMutilVNet2d():
    # 训练超参
    IMAGE_H, IMAGE_W = 512, 512   # 8GB VRAM; 上服务器改回 1024
    BATCH_SIZE = 2                 # 8GB VRAM; 上服务器改回 6
    EPOCHS = 150  # demo; 上服务器改回 500
    LR = 1e-3
    LOSS = 'MutilCrossEntropyDiceLoss'

    # wandb init
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"ziang-vnet2d-{IMAGE_H}x{IMAGE_W}-bs{BATCH_SIZE}",
        tags=["ziang", "vnet2d", "task1"],
        config={
            "architecture": "VNet2d",
            "image_size": f"{IMAGE_H}x{IMAGE_W}",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "loss": LOSS,
            "num_classes": 53,
            "dataset": "STS2024-2DXray",
            "gpu": "RTX5060-8GB",
        },
    )

    # Read data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\train2d.csv', header=None)
    maskdatasource = csvdata.iloc[:, 1].values
    imagedatasource = csvdata.iloc[:, 0].values
    # unlabel data is optional
    unlabel_csv = 'dataprocess\\data\\unlabel_train2d.csv'
    if os.path.exists(unlabel_csv) and os.path.getsize(unlabel_csv) > 0:
        csvdataaug = pd.read_csv(unlabel_csv, header=None)
        maskdataaug = csvdataaug.iloc[:, 1].values
        imagedataaug = csvdataaug.iloc[:, 0].values
        trainimages = np.concatenate((imagedatasource, imagedataaug), axis=0)
        trainlabels = np.concatenate((maskdatasource, maskdataaug), axis=0)
    else:
        trainimages = imagedatasource
        trainlabels = maskdatasource

    csv_data2 = pd.read_csv(r'dataprocess/data/valid2d.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    Vnet2d = MutilVNet2dModel(image_height=IMAGE_H, image_width=IMAGE_W, image_channel=1, numclass=53,
                              batch_size=BATCH_SIZE, loss_name=LOSS, accum_gradient_iter=1)
    Vnet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/dicece2d', epochs=EPOCHS, lr=LR)
    Vnet2d.clear_GPU_cache()
    wandb.finish()


if __name__ == '__main__':
    trainMutilVNet2d()
