# An example config file to be sent to wandb

import sys

sys.path.append("E:\BRATS_Research\BraTS-2021\src\model")

from wandb_creds import get_wandb_credentials
from loss import SoftDiceLossSquared
from ..model.spp_3d import Pyramid_Pooling_3D


import wandb
import torch


wandb.login(key=get_wandb_credentials(person="thejin"))


# Example sample Config
CONFIG = {
    "folds": 1,
    "epochs": 2,
    "LR": 1e-5,
    "Scehduler": "CosineAnnealling",
    "img_shape": 16,
    "model_conf": {
        "input_shape": (1, 32, [16, 16, 16]),
        "n_labels": 3,
        "activation": "relu",
        "normalizaiton": "group_normalizaiton",
        "c": 3,
    },
    "batch_size": 8,
    "SPP_info": {
        "used": True,
        "position": "Near bottleneck",  # Could include deep supervision pooling
    },
}

# wandb.init(project = 'Brats_21_Segmentation',config=CONFIG)
# wandb.run.name = "Sample run"
# wandb.run.save()

train_criterion = SoftDiceLossSquared()
out = torch.randn((4, 3, 128, 128, 128))
target = torch.randn((4, 3, 128, 128, 128))


batches = 3
# Simulated Training Loop
# net = NvNet(CONFIG["model_conf"])
net = Pyramid_Pooling_3D([2, 4, 6])
for i in range(CONFIG["folds"]):
    print(f"Fold {i} starting")
    run = wandb.init(project="Brats_21_Segmentation", config=CONFIG)
    wandb.run.name = f"Sample run fold {i}"
    wandb.run.save()
    for j in range(0, CONFIG["epochs"]):
        print(f"Epoch {j + 1} Started")
        for k in range(batches):
            out = torch.randn((4, 3, 128, 128, 128))
            target = torch.randn((4, 3, 128, 128, 128))
            loss = train_criterion(out, target)

        print(f"Saving weights for epoch {j}")

        torch.save(net.state_dict(), f"sample_weights/F{i}_E{j}_NVNET.pth")
        artifact = wandb.Artifact("model_weights", type="model")
        artifact.add_file(f"sample_weights/F{i}_E{j}_NVNET.pth")
        run.log_artifact(artifact)

        wandb.log(
            {
                "Epoch": j + 1,
                "Train Dice Loss": loss,
                "Validation Dice Loss": loss,
            }
        )
        print(f"Epoch {j + 1} done ")
    run.join()
    print(f"Fold {i} Ended")
