import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basic_blocks import (
    DownSampling,
    EncoderBlock,
    DecoderBlock,
    Deconvolution,
    AttentionBlock,
    OutputTransition,
)
from spp_block import Pyramid_Pooling_3D


# TODO: change the name of the architecture


class NvNet(nn.Module):
    def __init__(self, config, args):
        super(NvNet, self).__init__()

        self.config = config
        # some critical parameters
        self.inChans = config["input_shape"][1]
        self.input_shape = config["input_shape"]
        self.seg_outChans = config["output_channel"]

        # Encoder Blocks
        self.in_conv0 = DownSampling(
            inChans=self.inChans, outChans=32, stride=1, dropout_rate=0.2
        )
        self.en_block0 = EncoderBlock(32, 32, args)

        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64, args)
        self.en_block1_1 = EncoderBlock(64, 64, args)

        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128, args)
        self.en_block2_1 = EncoderBlock(128, 128, args)

        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256, args)
        self.en_block3_1 = EncoderBlock(256, 256, args)

        self.en_down4 = DownSampling(256, 512)
        self.en_block4_0 = EncoderBlock(512, 512, args)
        self.en_block4_1 = EncoderBlock(512, 512, args)
        self.en_block4_2 = EncoderBlock(512, 512, args)
        self.en_block4_3 = EncoderBlock(512, 512, args)

        ######################  SPATIAL PYRAMID POOLING BLOCK   ###########################
        # self.spp_inChans = self.en
        self.pyramid_pooling = Pyramid_Pooling_3D([2, 4, 8])
        self.bottleneck = nn.Conv3d(1024, 512, kernel_size=1)

        # Decoder Blocks
        # TODO try dilated convolutions
        self.de_up3 = Deconvolution(512, 256)
        self.de_block3 = DecoderBlock(256, 256, args)
        self.de_up2 = Deconvolution(256, 128)
        self.de_block2 = DecoderBlock(128, 128, args)
        self.de_up1 = Deconvolution(128, 64)
        self.de_block1 = DecoderBlock(64, 64, args)
        self.de_up0 = Deconvolution(64, 32)
        self.de_block0 = DecoderBlock(32, 32, args)

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
