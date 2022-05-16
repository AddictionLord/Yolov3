import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.insert(1, '/home/s200640/thesis/src/')
from blocks.cnn_block import CNNBlock




# Scale predictior outputs:
#   - number of channels = A * (N + 5)
#       --> A - anchor boxes (3)
#       --> N number of classes
#       --> 5 for [probability, x, y, w, h]
class ScalePrediction(nn.Module):
    def __init__(self, in_channels: int, num_of_classes: int):
        super(ScalePrediction, self).__init__()

        self.num_of_classes = num_of_classes
        self.block = nn.Sequential(
            CNNBlock(in_channels, in_channels * 2, kernel_size=3, padding=1),
            CNNBlock(in_channels * 2, 3 * (num_of_classes + 5), batch_norm=False, kernel_size=1)
        )


    # ------------------------------------------------------
    # W - Width, H - Height, B - batch_size, N - number of classes, A - number of anchor boxes
    # self.block(x) returns shape (B, A * (N + 5), W, H)
    # reshape it to (B, A, N + 5, W, H)
    # permute to (B, A, W, H, N + 5)
    # Doesn't really matter, format just needs to be consistent (for loss etc..)
    def forward(self, x: torch.tensor):

        return self.block(x).reshape(
            x.shape[0], 3, self.num_of_classes + 5, x.shape[2], x.shape[3]).permute(
                0, 1, 3, 4, 2
            )

