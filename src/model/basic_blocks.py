import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/joe-siyuan-qiao/WeightStandardization
# NOTE: apply this only for the conv layers before normalization.
class Conv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3, 4), keepdim=True)
        # https://github.com/joe-siyuan-qiao/WeightStandardization/issues/14#issuecomment-752319285
        # (
        #     weight.mean(dim=1, keepdim=True)
        #     .mean(dim=2, keepdim=True)
        #     .mean(dim=3, keepdim=True)
        # )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding)


class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(
        self, inChans, outChans, stride=2, kernel_size=3, padding=1, dropout_rate=None
    ):
        super(DownSampling, self).__init__()

        self.dropout_flag = False
        self.conv1 = nn.Conv3d(
            in_channels=inChans,
            out_channels=outChans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out


class EncoderBlock(nn.Module):
    """
    Encoder block
    """

    def __init__(
        self,
        inChans,
        outChans,
        args,
        stride=1,
        padding=1,
    ):
        super(EncoderBlock, self).__init__()

        self.activation = args.model.activation if args.model else args.activation
        self.normalization = (
            args.model.normalization if args.model else args.normalization
        )
        # https://amaarora.github.io/2020/08/09/groupnorm.html
        self.num_groups = args.model.num_groups if args.model else args.num_groups
        self.wt_std = args.model.wt_std if args.model else args.wt_std

        if self.normalization == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=inChans)
        else:
            # NOTE: Check for parameter 'affine'
            self.norm1 = nn.BatchNorm3d(num_features=inChans)
            self.norm2 = nn.BatchNorm3d(num_features=inChans)
        if self.activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif self.activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        elif self.activation == "leakyrelu":
            self.actv1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            self.actv2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if self.wt_std:
            self.conv1 = Conv3d(
                in_channels=inChans,
                out_channels=outChans,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
            self.conv2 = Conv3d(
                in_channels=inChans,
                out_channels=outChans,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = nn.Conv3d(
                in_channels=inChans,
                out_channels=outChans,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
            self.conv2 = nn.Conv3d(
                in_channels=inChans,
                out_channels=outChans,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        #         out = self.conv1(x)
        #         out = self.norm1(out)
        #         out = self.conv2(out)
        #         out = self.norm2(out)
        #         out = self.actv2(out)

        out += residual

        return out


class Deconvolution(nn.Module):
    def __init__(self, inChans, outChans, kernel_size=2, stride=2, dilation=0):
        super(Deconvolution, self).__init__()

        self.conv_t = nn.ConvTranspose3d(
            inChans, outChans, kernel_size, stride, dilation, bias=False
        )
        # to lower the dimension 512 -> 256
        self.conv_bottleneck = nn.Conv3d(inChans, outChans, kernel_size=1)

    def forward(self, x, skipx=None):

        out = self.conv_t(x)
        if skipx is not None:
            # print(out.shape, skipx.shape)
            # RuntimeError: torch.cat(): Sizes of tensors must match except in dimension 1. Got 1 and 2 in dimension 2 (The offending index is 1)
            # skipx = torch.reshape(skipx, (3, 256, 8, 1, 1))  # 2^3/1^2
            # soln2: kernel_size = 2
            out = torch.cat((out, skipx), 1)  # 1 -> 2 after above change
            out = self.conv_bottleneck(out)

        return out


class DecoderBlock(nn.Module):
    """
    Decoder block
    """

    def __init__(
        self,
        inChans,
        outChans,
        args,
        stride=1,
        padding=1,
    ):
        super(DecoderBlock, self).__init__()

        self.activation = args.model.activation if args.model else args.activation
        self.normalization = (
            args.model.normalization if args.model else args.normalization
        )
        # https://amaarora.github.io/2020/08/09/groupnorm.html
        self.num_groups = args.model.num_groups if args.model else args.num_groups
        self.wt_std = args.model.wt_std if args.model else args.wt_std

        if self.normalization == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=outChans)
        else:
            # NOTE: Check for parameter 'affine'
            self.norm1 = nn.BatchNorm3d(num_features=inChans)
            self.norm2 = nn.BatchNorm3d(num_features=inChans)
        if self.activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif self.activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        elif self.activation == "leakyrelu":
            self.actv1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            self.actv2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if self.wt_std:
            self.conv1 = Conv3d(
                in_channels=inChans,
                out_channels=outChans,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
            self.conv2 = Conv3d(
                in_channels=inChans,
                out_channels=outChans,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = nn.Conv3d(
                in_channels=inChans,
                out_channels=outChans,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
            self.conv2 = nn.Conv3d(
                in_channels=inChans,
                out_channels=outChans,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)

        #         out = self.conv1(x)
        #         out = self.norm1(out)
        #         out = self.actv1(out)
        #         out = self.conv2(out)
        #         out = self.norm2(out)
        #         out = self.actv2(out)

        out += residual

        return out


class OutputTransition(nn.Module):
    """
    Decoder output layer
    output the prediction of segmentation result
    """

    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels=inChans, out_channels=outChans, kernel_size=1
        )
        self.actv1 = torch.sigmoid
        """
        The softmax non-linearity at the final layer of the network was replaced by a
        sigmoid activation, treating each voxels as a multi-class classification problem.
        """

    def forward(self, x):
        return self.actv1(self.conv1(x))


class AttentionBlock(nn.Module):
    """
    stride = 2; please set --attention parameter to 1 when activate this block. the result name should include "att2".
    To fit in the structure of the V-net: F_l = F_int
    """

    def __init__(
        self, F_g, F_l, F_int, kernel_size=2, stride=2, scale_factor=2, mode="trilinear"
    ):
        super(AttentionBlock, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

        self.W_g = nn.Sequential(
            nn.Conv3d(
                F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True
            ),  # reduce num_channels
            nn.BatchNorm3d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(
                F_l,
                F_int,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=True,
            ),  # downsize
            nn.BatchNorm3d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, visualize=False):
        """
        :param g: gate signal from coarser scale
        :param x: the output of the l-th layer in the encoder
        :param visualize: enable this when plotting attention matrix
        :return:
        """
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        relu = self.relu(g1 + x1)
        sig = self.psi(relu)

        ################ Modifications possible ###############

        alpha = nn.functional.interpolate(
            sig, scale_factor=self.scale_factor, mode=self.mode
        )

        if visualize:
            return alpha
        else:
            return x * alpha
