import argparse, os, time, pathlib
from logging import warn
import datetime
import numpy as np
from termcolor import colored
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import wandb
from logger.wandb_creds import get_wandb_credentials

wandb.login(key=get_wandb_credentials(person="sanchit"))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from model.unet import rsppUnet
from loader.dataloader import get_dataset
from utils.losses import SoftDiceLossSquared
from utils.ops import seed_torch
from train import trainer


def main(args, fold_num):

    ngpus = torch.cuda.device_count()
    print(colored(f"Working with {ngpus} GPUs", "green"))
    print(colored("GPU Name: {}".format(torch.cuda.get_device_name(0)), "green"))

    seed_torch(args.seed)
    #     current_exp_time = datetime.now().strftime("%Y%m%d_%T").replace(":", "")
    current_time = datetime.datetime.now()
    subdir = (
        str(current_time.day)
        + "-"
        + str(current_time.month)
        + "-"
        + str(current_time.year)
        + "__"
    )
    time_name = (
        str(current_time.hour)
        + "_"
        + str(current_time.minute)
        + "_"
        + str(current_time.second)
    )
    current_exp_time = subdir + time_name
    print(colored("Current experiment time: {}".format(current_exp_time), "red"))
    exp_name = (
        f"{current_exp_time}_"
        f"_fold{(fold_num+1)}"
        f"_img{args.dataset.img_size[2]}"
        f"_optim"
        f"_{args.optimizer.optim}"
        f"_lr{args.optimizer.lr}_epochs{args.train.epochs}"
        f"{'_' + args.com.replace(' ', '_') if args.com else ''}"
    )

    wandb.init(project=args.logger.project_name, config=args)
    wandb.run.name = args.logger.run_name
    wandb.run.save()

    if args.save:
        save_folder = (
            f"/nfs/Workspace/brats_brain_segmentation/src/experiments/{exp_name}"
        )
        os.makedirs(save_folder, exist_ok=True)
        # print(save_folder)

    model_config = {
        "input_shape": (args.dataset.batch_size, 4, [args.dataset.img_size]),
        "output_channel": 3,
        "n_labels": 3,
        # "activation": args.model.activation,
        # "normalization": args.model.normalization,
    }

    model = rsppUnet(model_config, args)
    print(
        colored(
            f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
            "green",
        )
    )

    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(args.device)

    criterion = SoftDiceLossSquared().to(args.device)
    metric = None

    optimizer = Adam(
        model.parameters(), args.optimizer.lr, weight_decay=args.optimizer.wd, eps=1e-4
    )
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

        if args.save:
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


# if __name__ == "__main__":

#     arguments = parser.parse_args()
#     # os.environ["CUDA_VISIBLE_DEVICES"] = arguments.devices
#     #     for fold in range(1):
#     #         print(" ")
#     #         print(f"Fold {fold+1} statrting!!!")
#     main(arguments, 0)
