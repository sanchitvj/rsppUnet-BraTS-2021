import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    pooling dims -> [List] : [1,2,3]
        Contains the size of the pooling kernel. In total 2 pooling layers present. Each element in the 
        list corresponds to filter size n * n. Ex : Element at 0 index is 1 hence first pooling layer will
        have 1 *1 pooling size

    mode -> String : 'avg' or 'max'
        avg corresponds to average pooling
        max corresponds to max pooling
"""


class Pyramid_Pooling_3D(nn.Module):
    def __init__(self, levels, mode="max"):
        """
        Pyramid Pooling class uses the static spatial pyramid pooling method to calculate the pooling.
        :param levels -> [List] :  defines the filter size of the pooling layer to be used. Should be a list.
        :param mode   -> String : Decides whether to use max or average pooling. Should be either "max" or "avg"
        """
        super(Pyramid_Pooling_3D, self).__init__()
        # self.levels = levels
        assert len(levels) == 3

        if mode == 'max':
            self.pool_1 = nn.MaxPool3d((levels[2], levels[2], levels[2]))
            self.pool_2 = nn.MaxPool3d((levels[1], levels[1], levels[1]))
            self.pool_3 = nn.MaxPool3d((levels[0], levels[0], levels[0]))
        elif mode == 'avg':
            self.pool_1 = nn.AvgPool3d((levels[2], levels[2], levels[2]))
            self.pool_2 = nn.AvgPool3d((levels[1], levels[1], levels[1]))
            self.pool_3 = nn.AvgPool3d((levels[0], levels[0], levels[0]))

    def forward(self, en1, en2, en3):
        print(en1.shape)
        pooled_out_1 = self.pool_1(en1)
        pooled_out_2 = self.pool_2(en2)
        pooled_out_3 = self.pool_3(en3)
        cat = torch.cat((pooled_out_1, pooled_out_2, pooled_out_3), 1)
        return cat


if __name__ == '__main__':
    x_train = np.random.randn(3, 64, 16, 16, 16)
    x_train = torch.from_numpy(x_train).float()

    x_train_1 = np.random.randn(3, 128, 32, 32, 32)
    x_train_1 = torch.from_numpy(x_train_1).float()

    x_train_2 = np.random.randn(3, 256, 64, 64, 64)
    x_train_2 = torch.from_numpy(x_train_2).float()

    pyramid_pooling = Pyramid_Pooling_3D([2, 4, 8], 'max')
    out = pyramid_pooling(x_train_2, x_train_1, x_train)
    print(f"Output  shape {out.shape}")

"""
torch.Size([3, 32, 128, 128, 128]) -> 512, 64, 64, 64 => 576
torch.Size([3, 64, 64, 64, 64])    -> (3,64,4096,8,8
torch.Size([3, 128, 32, 32, 32]) -> (3,128,512,8,8)
torch.Size([3, 256, 16, 16, 16]) -> (3,256,64,8,8)
torch.Size([3, 512, 8, 8, 8]) -> As it is Concatenate the other 2 
"""

"""
Pooling size :
    Fpr 64*64*64 -> Pool size 8
    For 32*32*32 -> Pool size 4
    For 16*16*16 -> Pool size 2
"""



