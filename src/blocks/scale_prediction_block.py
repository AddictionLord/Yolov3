import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.insert(1, '/home/mary/thesis/project/src/')
from blocks.cnn_block import CNNBlock


# Scale predictior outputs:
#   - number of channels = 3 * (N + 5)
#       --> 3 anchor boxes
#       --> N number of classes
#       --> 5 for [probability, x, y, w, h]
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_of_classes):
        super(ScalePrediction, self).__init__()

        self.num_of_classes = num_of_classes
        self.block = nn.Sequential(
            CNNBlock(in_channels, in_channels * 2),
            CNNBlock(in_channels * 2, 3 * (num_of_classes + 5))            
        )


    # ------------------------------------------------------
    def forward(self, x):

        return self.block(x)

