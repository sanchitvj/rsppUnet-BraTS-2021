import yaml, random, os
import torch
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.spatial.distance import directed_hausdorff


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_args(args):
    """Save parsed arguments to config file."""
    config = vars(args).copy()
    del config["save_folder"]
    del config["seg_folder"]
    pprint.pprint(config)
    config_file = args.save_folder / (args.exp_name + ".yaml")
    with config_file.open("w") as file:
        yaml.dump(config, file)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def save_metrics(
    epoch, metrics, swa, writer, current_epoch, teacher=False, save_folder=None
):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(
        f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
        [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()],
    )
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(
            f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
            [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()],
            file=f,
        )
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{'_swa' if swa else ''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []
    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)
            haussdorf_dist = np.nan

        else:
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list


# from NVnet
def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):
    batch_size = targets.size(0)
    y_pred = outputs[:, 0, :, :, :]
    y_truth = targets[:, 0, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    intersection = torch.sum(torch.mul(y_pred, y_truth)) + eps / 2
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = 2 * intersection / union

    return dice / batch_size
