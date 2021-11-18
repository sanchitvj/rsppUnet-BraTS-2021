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
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        """
        super(Pyramid_Pooling_3D, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):

        """
        """
        num_sample = previous_conv.size(0)
        num_channel = previous_conv.size(1)
        previous_conv_size = [
                                int(previous_conv.size(-3)),
                                int(previous_conv.size(-2)),
                                int(previous_conv.size(-1))
                        ]

        for i in range(len(levels)):
            z_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            h_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[2] / levels[i]))

            z_pad1 = int(math.floor((z_kernel * levels[i] - previous_conv_size[0]) / 2))
            z_pad2 = int(math.ceil((z_kernel * levels[i] - previous_conv_size[0]) / 2))

            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[1]) / 2))

            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[2]) / 2))
            w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[2]) / 2))


            assert  w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[2]) and \
                    h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[1]) and \
                    z_pad1 + z_pad2 == (z_kernel * levels[i] - previous_conv_size[0])


            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2,z_pad1,z_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool3d((levels[i],levels[i],levels[i]))
            elif mode == "avg":
                pool = nn.AvgPool3d((z_kernel,h_kernel, w_kernel))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)

            dim = x.size(-1) **3
            x = x.numpy()
            by_factor = math.ceil(previous_conv_size[0]/levels[-1])
            dim /= math.ceil(previous_conv_size[0]/levels[-1])**2

            reshape_size = (num_sample, num_channel, math.ceil(dim), by_factor, by_factor)

            x = np.resize(x,reshape_size)
            if i == 0:
                spp = x
            else:
                spp = np.concatenate((spp,x),2)
        return spp

if __name__ == '__main__':
    x_train = np.random.randn(3, 1,64, 64, 64)
    x_train = torch.from_numpy(x_train).float()
    pyramid_pooling = Pyramid_Pooling_3D([2,4,6] , 'max')
    out = pyramid_pooling(x_train)
    print(f"Output  shape {out.shape}")







