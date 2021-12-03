import time, warnings

warnings.filterwarnings("ignore")

from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from ranger import Ranger
from torch.cuda.amp import autocast, GradScaler

from model.unet import NvNet
from loader.dataloader import get_dataset
from loss import SoftDiceLossSquared

from utils import AverageMeter, save_metrics, calculate_metrics, dice_coefficient

# Setting up logging stuff
import wandb

# Below function can also be used for validation
# INFO: https://discuss.pytorch.org/t/mixed-precision-in-evaluation/94184


def trainer(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    EPOCHS,
    epoch,
    phase,
    metric,
    device,
):
    print(" ")
    print(f"TRAINING EPOCH {epoch}")
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    # FIXME: accuracy meter
    accuracies = AverageMeter("Accuracy", ":2f")
    scaler = GradScaler()

    # TODO: get more info on perf_counter()
    start_point = time.perf_counter()

    metrics = []
    #     loader = tqdm(train_loader, total=len(train_loader))
    for i, batch in enumerate(train_loader):

        data_time.update(time.perf_counter() - start_point)
        # INFO: https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
        images = batch["image"].to(device)  # , non_blocking=True)
        labels = batch["mask"].to(device)

        # TODO: check floating point precision
        # NOTE: avg mixed precision will not affect validation
        with autocast():

            logits = model(images)
            loss = criterion(logits, labels)

            if not np.isnan(loss.item()):
                losses.update(loss.item())
            else:
                print("NaN in model loss!!")

            accuracy = dice_coefficient(
                logits.cpu(), labels.cpu(), threshold=0.5, eps=1e-8
            )
            accuracies.update(accuracy, batch["image"].size(0))

        #             met_ = calculate_metrics(logits, labels, None)
        #             print(met_)

        #             if not model.training:
        #                 metric_ = metric(logits, labels)
        #                 metrics.extend(metric_)

        if model.training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)

        #         loader.set_description(f"{phase} Epoch {epoch+1}/{EPOCHS}")
        #         loader.set_postfix(loss=losses.avg, accuracy=accuracies.avg)

        if scheduler is not None:
            scheduler.step()

        batch_time.update(time.perf_counter() - start_point)
        start_point = time.perf_counter()

        # For Logging on a batch basis

        # if phase == "Training":
        #     wandb.log(
        #         {
        #             "Train Dice Loss": loss,
        #             "Train Class 0 Dice": metric_[0][0],
        #             "Train Class 1 Dice": metric_[0][1],
        #             "Train Class 2 Dice": metric_[0][2],
        #         }
        #     )
        # elif phase == "Validating":
        #     wandb.log(
        #         {
        #             "Valid Dice Loss": loss,
        #             "Valid Class 0 Dice": metric_[0][0],
        #             "Valid Class 1 Dice": metric_[0][1],
        #             "Valid Class 2 Dice": metric_[0][2],
        #         }
        #     )

    # TODO: add wandb logger

    # Remove this comment for logging on epoch basis
    if phase == "Training":
        wandb.log({"Train Dice Loss": losses.avg})
        wandb.log({"Train Dice Score": accuracies.avg})
    elif phase == "Validating":
        wandb.log({"Validation Dice Loss": losses.avg})
        wandb.log({"Validation Dice Score": accuracies.avg})

    # if not model.training:
    # save_metrics(epoch, metrics, swa, logger, epoch, False, save_folder)

    return losses.avg, accuracies.avg


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     image = out = torch.randn((2, 3, 128, 128, 128))
#     target = torch.randn((2, 3, 128, 128, 128))
#     config = {
#         "input_shape": (1, 32, [128, 128, 128]),
#         "n_labels": 3,
#         "activation": "relu",
#         "normalizaiton": "group_normalizaiton",
#     }
#     net = NvNet(config).to(device)
#     parameters = params = filter(lambda p: p.requires_grad, net.parameters())
#     optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#     criterion = SoftDiceLossSquared()
# print(
#     testing_train_loop(image, target, net, optimizer, criterion, scheduler, device)
# )
