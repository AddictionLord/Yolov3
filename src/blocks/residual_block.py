import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.insert(1, '/home/mary/thesis/project/src/')
from utils.weights_handler import WeightsHandler
from blocks.cnn_block import CNNBlock




# ------------------------------------------------------  
#   
# ------------------------------------------------------    
class ResidualBlock(nn.Module):
    def __init__(self, num_of_repeats, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential()
        res_block = nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=1, padding=0),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1)
        )

        for repeat in range(num_of_repeats):

            self.block = nn.Sequential(*self.block, *res_block)


    # ------------------------------------------------------
    def forward(self, x):

        return self.block(x)


    # ------------------------------------------------------
    def loadWeights(self, weights: WeightsHandler):

        for cnn_block in self.block:

            cnn_block.loadWeights(weights)






# ------------------------------------------------------
# Main - mostly for testing purposes
# ------------------------------------------------------
if __name__ == '__main__':


    # t = torch.rand(1, 3, 256, 256)
    # kernel_size = 3
    # padding = 1 if kernel_size == 3 else 0

    # pre = nn.Sequential(
    #     CNNBlock(3, 32, 3, stride=1, padding=padding),
    #     CNNBlock(32, 64, 3, stride=2, padding=padding)
    # )
    # out = pre(t)
    # print(out.shape)

    # res = ResidualBlock(1, 64)
    # out = res(out)
    # print(out.shape)    

    # --------------------------------
    darknet53_path = './pretrained/darknet53.conv.74'
    w = WeightsHandler(darknet53_path)

    res = ResidualBlock(4, 3)
    print(res.block[0].block[0].weight[0, 2, 0, 0])
    res.loadWeights(w)
    print(res.block[0].block[0].weight[0, 2, 0, 0])

