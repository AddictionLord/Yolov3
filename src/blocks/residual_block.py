import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.insert(1, '/home/mary/thesis/project/src/')
from utils import WeightsHandler
from blocks import CNNBlock




class ResidualBlock(nn.Module):
    def __init__(self, num_of_repeats: int, in_channels: int, residual=True):
        super(ResidualBlock, self).__init__()
        
        self.residual = residual
        self.num_of_repeats = num_of_repeats
        self.block = nn.ModuleList()
        res_block = nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=1, padding=0),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1)
        )

        for repeat in range(num_of_repeats):

            # self.block += nn.Sequential(res_block)
            self.block.append(res_block)


    # ------------------------------------------------------
    def forward(self, x: torch.tensor):

        for layer in self.block:

            if self.residual:
                x += layer(x)

            else:
                x = layer(x)

        return x


    # ------------------------------------------------------
    def loadWeights(self, weights: WeightsHandler):

        for sequential_block in self.block:
            for cnn_block in sequential_block:

                cnn_block.loadWeights(weights)






# ------------------------------------------------------
# Main - mostly for testing purposes
# ------------------------------------------------------
if __name__ == '__main__':


    # t = torch.rand(1, 3, 256, 256)
    # kernel_size = 3
    # padding = 1 if kernel_size == 3 else 0

    # pre = nn.Sequential(
    #     CNNBlock(3, 32, kernel_size=3, stride=1, padding=padding),
    #     CNNBlock(32, 64, kernel_size=3, stride=2, padding=padding)
    # )
    # out = pre(t)
    # print(out.shape)

    # res = ResidualBlock(1, 64, False)
    # out = res(out)
    # print(out.shape)    

    # --------------------------------
    t = torch.rand(1, 64, 256, 256)

    res = ResidualBlock(1, 64)
    out = res(t)

    print(out.shape)


    # --------------------------------
    darknet53_path = './pretrained/darknet53.conv.74'
    w = WeightsHandler(darknet53_path)

    res = ResidualBlock(4, 3)
    print(res.block[0][0].block[0].weight[0, 2, 0, 0])
    res.loadWeights(w)
    print(res.block[0][0].block[0].weight[0, 2, 0, 0])

