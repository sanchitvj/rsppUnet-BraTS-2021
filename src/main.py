import argparse, os, time, pathlib
from logging import warn
from datetime import datetime
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import wandb
from logger.wandb_creds import get_wandb_credentials

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from model.unet import NvNet
from loader.dataloader import get_dataset
from loss import EDiceLoss, SoftDiceLossSquared
from utils import save_args, seed_torch, determinist_collate
from train import trainer

# from utils import AverageMeter,

parser = argparse.ArgumentParser(description="BraTS Training")

parser.add_argument(
    "-d",
    "--device",
    required=True,
    type=str,
    default="cpu",
    help="Set the CUDA_VISIBLE_DEVICES env var from this string",
)
parser.add_argument(
    "-p",
    "--data_path",
    default="/nfs/Workspace/brats_brain_segmentation/data/BraTS2020_data/training",
)
parser.add_argument(
    "-t",
    "--teacher_model",
    default=False,
    help="Make it true for training on 2020 dataset",
)
# INFO: https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
parser.add_argument(
    "-j",
    "--num_workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 16).",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--epochs", default=15, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 1)",
)
parser.add_argument(
    "-i",
    "--img_size",
    default=(160, 160, 160),
    type=int,
    help="image size (default: (128,128,128))",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.00025,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-7,
    type=float,
    metavar="W",
    help="weight decay (default: 0)",
    dest="weight_decay",
)
# Warning: untested option!!
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint. Warning: untested option",
)
parser.add_argument("--debug", action="store_true", default=False)
# parser.add_argument("--no_fp16", action="store_true")
parser.add_argument("--seed", default=12536, help="seed for train/val split")
parser.add_argument("--warm", default=3, type=int, help="number of warming up epochs")

parser.add_argument(
    "--val", default=3, type=int, help="how often to perform validation step"
)
parser.add_argument("--fold", default=0, type=int, help="Split number (0 to 4)")
parser.add_argument("--activation", default="relu")
# TODO: experiment with normalization
parser.add_argument("--normalization", default="group_normalization")
parser.add_argument(
    "--swa",
    action="store_true",
    help="perform stochastic weight averaging at the end of the training",
)
parser.add_argument(
    "--swa_repeat", type=int, default=5, help="how many warm restarts to perform"
)
parser.add_argument(
    "--optim", choices=["adam", "sgd", "ranger", "adamw"], default="adam"
)
parser.add_argument("--com", help="add a comment to this run!")
parser.add_argument(
    "--dropout", type=float, help="amount of dropout to use", default=0.0
)
parser.add_argument(
    "--warm_restart",
    action="store_true",
    help="use scheduler warm restarts with period of 30",
)
parser.add_argument(
    "--full", action="store_true", help="Fit the network on the full training set"
)


def main(args, fold_num):

    wandb.login(key=get_wandb_credentials(person="sanchit"))

    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")
    print("GPU Name: ", torch.cuda.get_device_name(0))
    seed_torch(args.seed)
    current_exp_time = datetime.now().strftime("%Y%m%d_%T").replace(":", "")
    print("Current experiment time: ", current_exp_time)
    args.exp_name = (
        f"{'debug_' if args.debug else ''}{current_exp_time}_"
        f"_fold{fold_num if not args.full else 'FULL'}"
        f"_batch{args.batch_size}"
        f"_optim{args.optim}"
        f"_{args.optim}"
        f"_lr{args.lr}-wd{args.weight_decay}_epochs{args.epochs}"
        # f"_{'fp16' if not args.no_fp16 else 'fp32'}"
        f"_warm{args.warm}_"
        f"_norm{args.normalization}{'_swa' + str(args.swa_repeat) if args.swa else ''}"
        f"_dropout{args.dropout}"
        f"_warm_restart{args.warm_restart}"
        f"{'_' + args.com.replace(' ', '_') if args.com else ''}"
    )

    args.save_folder = pathlib.Path(f"./experiments/{args.exp_name}")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    args.save_folder = args.save_folder.resolve()
    save_args(args)

    # NOTE: Logger
    wandb.init(project=f"BraTS", config=args.exp_name)

    model_config = {
        "input_shape": (args.batch_size, 4, [args.img_size]),
        "output_channel": 3,
        "n_labels": 3,
        "activation": args.activation,
        "normalization": args.normalization,
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

    criterion = SoftDiceLossSquared().to(args.device)
    #     metric = criterion.metric
    metric = None
    # print(metric)

    # TODO: play with optimizers
    optimizer = Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay, eps=1e-4
    )
    # TODO: add warm up and experiment with parameters
    scheduler = CosineAnnealingLR(optimizer, args.epochs + round(args.epochs * 0.5))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume)
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f">>>> Checkpoint loaded from epoch {start_epoch+1} <<<<")

    train_dataset, val_dataset = get_dataset(
        args.data_path,
        args.seed,
        args.img_size,
        fold_num,
        args.teacher_model,
        args.debug,
        n_splits=5,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # INFO: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
        drop_last=True,
        # INFO: https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/23
        # collate_fn=determinist_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        # collate_fn=determinist_collate,
    )

    best = np.inf

    for epoch in range(start_epoch, args.epochs):

        model.train()
        t_loss, t_acc = trainer(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            args.epochs,
            epoch,
            "Training",
            metric,
            args.device,
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
                args.epochs,
                epoch,
                "Validating",
                metric,
                args.device,
            )
            print(f"Validation loss at the end of epoch {epoch+1} is: {v_loss}")
            print(f"Validation accuracy at the end of epoch {epoch+1} is: {v_acc}")

        if v_loss < best:
            best = v_loss
            torch.save(
                dict(
                    epoch=epoch,
                    model=model,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                ),
                f"/nfs/Workspace/brats_brain_segmentation/save_folder/model_{v_acc}_fold{fold_num}.pth",
            )

        if epoch / args.epochs > 0.5:
            scheduler.step()
            print("scheduler stepped!")

    wandb.run.name = f"BRATS_RUN_F{fold_num+1}"
    wandb.run.save()
    # TODO: generate segmentation maps


if __name__ == "__main__":

    arguments = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = arguments.devices
    for fold in range(5):
        print(" ")
        print(f"Fold {fold+1} statrting!!!")
        main(arguments, fold)
