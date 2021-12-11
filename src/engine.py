import argparse, os, time, pathlib
from logging import warn
from datetime import datetime
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import wandb
from logger.wandb_creds import get_wandb_credentials

wandb.login(key=get_wandb_credentials(person="sanchit"))

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from model.unet import NvNet
from loader.dataloader import get_dataset
from loss import EDiceLoss, SoftDiceLossSquared
from utils import save_args, seed_torch, determinist_collate
from train import trainer


def main(args, fold_num):

    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")
    print("GPU Name: ", torch.cuda.get_device_name(0))

    seed_torch(args.seed)
    current_exp_time = datetime.now().strftime("%Y%m%d_%T").replace(":", "")
    print("Current experiment time: ", current_exp_time)
    exp_name = (
        f"{current_exp_time}_"
        f"_fold{(fold_num+1)}"
        f"_batch{args.dataset.batch_size}"
        f"_img{args.dataset.img_size[0]}"
        f"_optim"
        f"_{args.optimizer.optim}"
        f"_lr{args.optimizer.lr}_epochs{args.train.epochs}"
        # f"_norm{args.model.normalization}"
        f"{'_' + args.com.replace(' ', '_') if args.com else ''}"
    )
    wandb.init(project=args.logger.project_name, config=args)

    save_folder = f"/nfs/Workspace/brats_brain_segmentation/src/experiments/{exp_name}"
    os.makedirs(save_folder, exist_ok=True)
    # print(save_folder)

    model_config = {
        "input_shape": (args.dataset.batch_size, 4, [args.dataset.img_size]),
        "output_channel": 3,
        "n_labels": 3,
        "activation": args.model.activation,
        "normalization": args.model.normalization,
    }

    # TODO: change the name of the architecture
    model = NvNet(model_config)
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # TODO: try more than one GPU
    # if ngpus > 1:
    #     model = torch.nn.DataParallel(model).cuda()
    # else:
    model = model.to(args.device)

    # TODO: experiment with losses
    criterion = SoftDiceLossSquared().to(args.device)
    #     metric = criterion.metric
    metric = None
    # print(metric)

    # TODO: play with optimizers
    optimizer = Adam(
        model.parameters(), args.optimizer.lr, weight_decay=args.optimizer.wd, eps=1e-4
    )
    # TODO: add warm up and experiment with parameters
    scheduler = CosineAnnealingLR(
        optimizer, args.train.epochs + round(args.train.epochs * 0.5)
    )

    start_epoch = 0
    if args.train.resume is not None:
        ckpt = torch.load(args.train.resume)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f">>>> Checkpoint loaded from epoch {start_epoch} <<<<")

    # augs = True
    train_dataset, val_dataset = get_dataset(args, fold_num, n_splits=5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.dataset.batch_size,
        shuffle=True,
        num_workers=args.dataset.num_workers,
        pin_memory=True,  # INFO: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
        drop_last=True,
        # INFO: https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/23
        # collate_fn=determinist_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.dataset.batch_size,
        shuffle=False,
        num_workers=args.dataset.num_workers,
        pin_memory=False,
        drop_last=False,
        # collate_fn=determinist_collate,
    )

    best = np.inf
    if args.debug:
        args.train.epochs = 2
    for epoch in range(start_epoch, args.train.epochs):

        model.train()
        t_loss, t_acc = trainer(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            args.train.epochs,
            epoch,
            "Training",
            metric,
            args.device,
            args.debug,
        )
        print(f"Training loss at the end of epoch {epoch+1} is: {t_loss}")
        print(f"Training accuracy at the end of epoch {epoch+1} is: {t_acc}")

        # NOTE: validating after every epoch
        model.eval()
        with torch.no_grad():
            v_loss, v_acc = trainer(
                val_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                args.train.epochs,
                epoch,
                "Validating",
                metric,
                args.device,
                args.debug,
            )
            print(f"Validation loss at the end of epoch {epoch+1} is: {v_loss}")
            print(f"Validation accuracy at the end of epoch {epoch+1} is: {v_acc}")

        if v_loss < best:
            best = v_loss
            torch.save(
                dict(
                    epoch=epoch + 1,
                    model=model,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                ),
                f"{save_folder}/model_{v_acc:.3f}_{v_loss:.3f}.pth",
            )

        if epoch / args.train.epochs > 0.5:
            scheduler.step()
            print("scheduler stepped!")

    wandb.run.name = f"BRATS_F{fold_num+1}"
    wandb.run.save()
    # TODO: generate segmentation maps


# if __name__ == "__main__":

#     arguments = parser.parse_args()
#     # os.environ["CUDA_VISIBLE_DEVICES"] = arguments.devices
#     #     for fold in range(1):
#     #         print(" ")
#     #         print(f"Fold {fold+1} statrting!!!")
#     main(arguments, 0)
