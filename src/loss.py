import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


# TODO: add metrics mode in Soft Dl.
# Taken from NNUnet
class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.0):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, inputs, targets, loss_mask=None):
        shp_x = inputs.shape
        shp_y = targets.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            inputs = self.apply_nonlin(inputs)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                targets = targets.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(inputs.shape, targets.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = targets
            else:
                targets = targets.long()
                y_onehot = torch.zeros(shp_x)
                if inputs.device.type == "cuda":
                    y_onehot = y_onehot.cuda(inputs.device.index)
                y_onehot.scatter_(1, targets, 1).float()

        intersect = inputs * y_onehot
        # values in the denominator get smoothed
        denominator = inputs ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        # dc -> 1-dc
        return 1 - dc


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need."""

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.0
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                # print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1.0, device="cuda")
                else:
                    return torch.tensor(0.0, device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (
                inputs.pow(2).sum() + targets.pow(2).sum() + smooth
            )
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        # TODO: check alpha, gamma, smooth
        alpha=0.5,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DC_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs):
        super(DC_and_Focal_loss, self).__init__()

        softmax_helper = lambda x: F.softmax(x, 1)

        # TODO: check the difference between SDL and SDL squared.
        self.dc = SoftDiceLossSquared(**soft_dice_kwargs)
        self.focal = FocalLoss(**focal_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        focal_loss = self.focal(net_output, target)

        result = dc_loss + focal_loss
        return result


# if __name__ == "__main__":
#     loss = SoftDiceLossSquared()
#     out = torch.randn((4, 3, 128, 128, 128))
#     target = torch.randn((4, 3, 128, 128, 128))
#     print(loss(out, target))

#     brats_2020_loss = EDiceLoss()
#     print(brats_2020_loss(out, target))
