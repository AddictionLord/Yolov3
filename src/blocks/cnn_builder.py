import torch
import torch.nn as nn

from blocks.cnn_block import CNNBlock
from blocks.residual_block import ResidualBlock
from blocks.scale_prediction_block import ScalePrediction



darknet_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["R", 1],
    (128, 3, 2),
    ["R", 2],
    (256, 3, 2),
    ["R", 8],
    (512, 3, 2),
    ["R", 8],
    (1024, 3, 2),
    ["R", 4]
]

yolo_config = [
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S"
]


class CNNBuilder:

    # ------------------------------------------------------
    def _constructNeuralNetwork(self, config):

        in_channels = self.in_channels
        net = nn.ModuleList()
        for block in self.config:

            # Construction of CNNBlock and integration to network
            # CNNBlock changes number of channels - update:
            if isinstance(block, tuple):
                out_channels, kernel_size, stride = block
                layer = self._buildCNNLayer(in_channels, block)
                in_channels = out_channels

            # Construction of ResidualBlock and integration to network
            # ResidualBlock doesn't change number of channels (no update needed)
            elif isinstance(block, list):
                block_type, num_of_repeats = block
                layer = self._buildResidualLayer(in_channels, block)

            net.append(layer)

        return net



    # ------------------------------------------------------
    def _buildCNNLayer(self, in_channels, block):

        out_channels, kernel_size, stride = block
        layer = CNNBlock(
            in_channels, 
            out_channels,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=1 if kernel_size == 3 else 0
        )

        return layer


    # ------------------------------------------------------
    def _buildResidualLayer(self, in_channels, block):

        block_type, num_of_repeats = block
        layer = ResidualBlock(
            num_of_repeats, 
            in_channels
        )

        return layer


    def _buildScalePredictionLayer(self, in_channels, block):

        pass


