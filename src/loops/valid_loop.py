
import sys

sys.path.append('..')
sys.path.append('../model/')
sys.path.append('../loss/')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from model.spp_3d import *
from model.unet import NvNet
from loss.loss import SoftDiceLossSquared
sys.path.append('..')
from dataloader import *

import torch
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

BATCH_SIZE = 4



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def valid_loop( model, loader, criterion,device):
    losses = AverageMeter()
    scaler = GradScaler()
    model.to(device)
    model.eval()

    loader = tqdm(loader,total = len(loader))

    with torch.no_grad():
        for step, batch in enumerate(loader):
            image = batch['image'].to(device).float()
            seg_truth = batch['mask'].to(device)

            seg_out = model(image)
            loss = criterion(seg_out, seg_truth)
            losses.update(loss,BATCH_SIZE)
        
    return losses.avg

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        "input_shape": (1, 4, [128, 128, 128]),
        "n_labels": 3,
        "activation": "relu",
        "normalizaiton": "group_normalizaiton",
    }

    train_dir = "/nfs/Workspace/brats_brain_segmentation/data/BraTS2021_data/training"

    train_dataset, val_dataset = get_dataset(train_dir, 1111, fold_num=2)

    validation_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16,pin_memory = True
        )

    net = NvNet(config)

    criterion = SoftDiceLossSquared()
    print(f"Sample Validation Loss : {valid_loop( net,validation_loader, criterion, device)}")





