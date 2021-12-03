import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .spp_3d import Pyramid_Pooling_3D


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
        stride=1,
        padding=1,
        num_groups=8,
        activation="relu",
        normalization="batch",
    ):
        super(EncoderBlock, self).__init__()

        # self.norm1 = 0
        # TODO: try different normalizations
        if normalization == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
        else:
            # NOTE: Check for parameter 'affine'
            self.norm1 = nn.BatchNorm3d(num_features=inChans)
            self.norm2 = nn.BatchNorm3d(num_features=inChans)
        # print(self.norm1)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
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
        stride=1,
        padding=1,
        num_groups=8,
        activation="relu",
        normalization="group_normalization",
    ):
        super(DecoderBlock, self).__init__()

        if normalization == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
        else:
            # NOTE: Check for parameter 'affine'
            self.norm1 = nn.BatchNorm3d(num_features=inChans)
            self.norm2 = nn.BatchNorm3d(num_features=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_channels=inChans,
            out_channels=outChans,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )
        self.conv2 = nn.Conv3d(
            in_channels=outChans,
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


# TODO: change the name of the architecture


class NvNet(nn.Module):
    def __init__(self, config):
        super(NvNet, self).__init__()

        self.config = config
        # some critical parameters
        self.inChans = config["input_shape"][1]
        self.input_shape = config["input_shape"]
        self.seg_outChans = config["output_channel"]
        self.activation = config["activation"]
        self.normalization = config["normalization"]

        # Encoder Blocks
        self.in_conv0 = DownSampling(
            inChans=self.inChans, outChans=32, stride=1, dropout_rate=0.2
        )
        self.en_block0 = EncoderBlock(
            32, 32, activation=self.activation, normalization=self.normalization
        )

        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(
            64, 64, activation=self.activation, normalization=self.normalization
        )
        self.en_block1_1 = EncoderBlock(
            64, 64, activation=self.activation, normalization=self.normalization
        )

        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(
            128, 128, activation=self.activation, normalization=self.normalization
        )
        self.en_block2_1 = EncoderBlock(
            128, 128, activation=self.activation, normalization=self.normalization
        )

        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(
            256, 256, activation=self.activation, normalization=self.normalization
        )
        self.en_block3_1 = EncoderBlock(
            256, 256, activation=self.activation, normalization=self.normalization
        )

        self.en_down4 = DownSampling(256, 512)
        self.en_block4_0 = EncoderBlock(
            512, 512, activation=self.activation, normalization=self.normalization
        )
        self.en_block4_1 = EncoderBlock(
            512, 512, activation=self.activation, normalization=self.normalization
        )
        self.en_block4_2 = EncoderBlock(
            512, 512, activation=self.activation, normalization=self.normalization
        )
        self.en_block4_3 = EncoderBlock(
            512, 512, activation=self.activation, normalization=self.normalization
        )

        ######################  SPATIAL PYRAMID POOLING BLOCK   ###########################
        # self.spp_inChans = self.en
        self.pyramid_pooling = Pyramid_Pooling_3D([2, 4, 8])
        self.bottleneck = nn.Conv3d(1024, 512, kernel_size=1)

        # Decoder Blocks
        # TODO try dilated convolutions
        self.de_up3 = Deconvolution(512, 256)
        self.de_block3 = DecoderBlock(
            256, 256, activation=self.activation, normalization=self.normalization
        )
        self.de_up2 = Deconvolution(256, 128)
        self.de_block2 = DecoderBlock(
            128, 128, activation=self.activation, normalization=self.normalization
        )
        self.de_up1 = Deconvolution(128, 64)
        self.de_block1 = DecoderBlock(
            64, 64, activation=self.activation, normalization=self.normalization
        )
        self.de_up0 = Deconvolution(64, 32)
        self.de_block0 = DecoderBlock(
            32, 32, activation=self.activation, normalization=self.normalization
        )

        # Attention Blocks
        self.ag_3 = AttentionBlock(512, 256, 256)  # forward(g, x)
        self.ag_2 = AttentionBlock(256, 128, 128)
        self.ag_1 = AttentionBlock(128, 64, 64)
        self.ag_0 = AttentionBlock(64, 32, 32)

        self.de_end = OutputTransition(32, self.seg_outChans)

    def forward(self, x):
        out_init = self.in_conv0(x)  # (7, 128, 192, 160)
        out_en0 = self.en_block0(out_init)
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0)))
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))
        out_en3 = self.en_block3_1(self.en_block3_0(self.en_down3(out_en2)))
        out_en4 = self.en_block4_3(
            self.en_block4_2(self.en_block4_1(self.en_block4_0(self.en_down4(out_en3))))
        )
        out_spp3d = self.pyramid_pooling(out_en1, out_en2, out_en3)

        enc_spp_out = torch.cat((out_en4, out_spp3d), 1)
        enc_spp_out = self.bottleneck(enc_spp_out)

        out_de3 = self.de_block3(self.de_up3(out_en4, self.ag_3(enc_spp_out, out_en3)))
        out_de2 = self.de_block2(self.de_up2(out_de3, self.ag_2(out_de3, out_en2)))
        out_de1 = self.de_block1(self.de_up1(out_de2, self.ag_1(out_de2, out_en1)))
        out_de0 = self.de_block0(self.de_up0(out_de1, self.ag_0(out_de1, out_en0)))

        out_end = self.de_end(out_de0)

        return out_end


# if __name__ == "__main__":

#     x_train = np.random.randn(3, 32, 128, 128, 128)
#     x_train = torch.from_numpy(x_train).float()

#     config = {
#         "input_shape": (1, 32, [16, 16, 16]),
#         "c" : 3,
#         "n_labels": 3,
#         "activation": "relu",
#         "normalization": "group_normalization",
#     }
#     net = NvNet(config)
#     out = net(x_train)
#     print(f"Output shape {out.shape}")
# Output  shape torch.Size([3, 3, 16, 16, 16])
