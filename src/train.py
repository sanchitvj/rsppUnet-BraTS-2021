import time, warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler

from utils.metrics import (
    AverageMeter,
    dice_coefficient,
    dice_per_region,
)

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
    device,
    debug,
):
    print(" ")
    print(f"*****  {phase} epoch {epoch+1} *****")
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    accuracies = AverageMeter("Accuracy", ":2f")
    dice_wts, dice_tcs, dice_ets, hd_ets, hd_tcs, hd_wts = (
        AverageMeter("WT score", ":2f"),
        AverageMeter("TC score", ":2f"),
        AverageMeter("ET score", ":2f"),
        AverageMeter("HD WT score", ":2f"),
        AverageMeter("HD TC score", ":2f"),
        AverageMeter("HD ET score", ":2f"),
    )
    dice_scores = AverageMeter("Dice Score", ":2f")
    hds = AverageMeter("Avg. Hausdorff Distance", ":2f")

    scaler = GradScaler()
    start_point = time.perf_counter()

    if debug:
        train_loader = tqdm(train_loader, total=len(train_loader))
    for i, batch in enumerate(train_loader):

        data_time.update(time.perf_counter() - start_point)
        # INFO: https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
        images = batch["image"].to(device, non_blocking=True, dtype=torch.float)
        labels = batch["mask"].to(device, dtype=torch.float)

        # NOTE: avg mixed precision will not affect validation
        with autocast():

            logits = model(images)
            loss = criterion(logits, labels)

            if not np.isnan(loss.item()):
                losses.update(loss.item())
            else:
                print("NaN in model loss!!")

            with torch.no_grad():
                accuracy = dice_coefficient(
                    logits.cpu(), labels.cpu(), threshold=0.5, eps=1e-8
                )

                accuracies.update(accuracy.detach().numpy(), batch["image"].size(0))

                if phase == "Validating":

                    dice_et, dice_tc, dice_wt, hd_et, hd_tc, hd_wt = dice_per_region(
                        logits.cpu(), labels.cpu()
                    )
                    dice_ets.update(dice_et.detach().numpy(), batch["image"].size(0))
                    dice_tcs.update(dice_tc.detach().numpy(), batch["image"].size(0))
                    dice_wts.update(dice_wt.detach().numpy(), batch["image"].size(0))
                    hd_ets.update(hd_et, batch["image"].size(0))
                    hd_tcs.update(hd_tc, batch["image"].size(0))
                    hd_wts.update(hd_wt, batch["image"].size(0))

                    dice_score = (dice_et + dice_tc + dice_wt) / 3
                    dice_scores.update(
                        dice_score.detach().numpy(), batch["image"].size(0)
                    )
                    hd = (hd_et + hd_tc + hd_wt) / 3
                    hds.update(hd, batch["image"].size(0))

        if model.training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if debug:
            if model.training:
                train_loader.set_description(f"{phase} Epoch {epoch+1}/{EPOCHS}")
                train_loader.set_postfix(
                    loss=losses.avg,
                    accuracy=accuracies.avg,
                    # wt=dice_wts.avg,
                    # tc=dice_tcs.avg,
                    # et=dice_ets.avg,
                    # hdwt=hd_wts.avg,
                    # hdtc=hd_tcs.avg,
                    # hdet=hd_ets.avg,
                    # ds=dice_scores.avg,
                )

            else:
                train_loader.set_description(f"{phase} Epoch {epoch+1}/{EPOCHS}")
                train_loader.set_postfix(
                    loss=losses.avg,
                    accuracy=accuracies.avg,
                    # wt=dice_wts.avg,
                    # tc=dice_tcs.avg,
                    # et=dice_ets.avg,
                    wt=hd_wts.avg,
                    tc=hd_tcs.avg,
                    et=hd_ets.avg,
                    ds=dice_scores.avg,
                )

        if scheduler is not None:
            scheduler.step()

        batch_time.update(time.perf_counter() - start_point)
        start_point = time.perf_counter()

    # Remove this comment for logging on epoch basis
    if phase == "Training":
        wandb.log({"Train Dice Loss": losses.avg})
        wandb.log({"Train Accuracy": accuracies.avg})
        score = accuracies.avg
    elif phase == "Validating":
        wandb.log(
            {
                "Validation Dice Loss": losses.avg,
                "Validation Accuracy": accuracies.avg,
                "Validation Dice Score": dice_scores.avg,
                "Whole Tumor Dice Score": dice_wts.avg,
                "Enhancing Tumor Dice Score": dice_ets.avg,
                "Tumor Core Dice Score": dice_tcs.avg,
                "Whole Tumor Hausdorff Distance": hd_wts.avg,
                "Enhancing Tumor Hausdorff Distance": hd_ets.avg,
                "Tumor Core Hausdorff Distance": hd_tcs.avg,
                "Avg. Hausdorff Distance": hds.avg,
            }
        )
        score = dice_scores.avg

    return losses.avg, score


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
#     net = rsppUnet(config).to(device)
#     parameters = params = filter(lambda p: p.requires_grad, net.parameters())
#     optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
#     criterion = SoftDiceLossSquared()
# print(
#     testing_train_loop(image, target, net, optimizer, criterion, scheduler, device)
# )
