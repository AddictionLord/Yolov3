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
'''
Set: {"U", 2} - means upsampling layer, scale_factor=2
List: ["R", 8] - means residual layer, number_of_repeats=8
      ["C", 8] - means convolutinal layers, number_of_repeats=8
Strign: "S" - means scale prediction layer
'''
yolo = [
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
    ["R", 4], #darknet53 to this point
    (512, 1, 1),
    (1024, 3, 1),
    ["C", 1],
    (512, 1, 1),
    "S",
    (256, 1, 1),
    {"U": 2},
    (256, 1, 1),
    (512, 3, 1),
    ["C", 1],
    (256, 1, 1),
    "S",
    (128, 1, 1),
    {"U": 2},
    (128, 1, 1),
    (256, 3, 1),
    ["C", 1],
    (128, 1, 1),
    "S"
]

class CNNBuilder:
    def __init__(self, in_channels, num_of_classes, config=yolo):

        self.in_channels = in_channels
        self.num_of_classes = num_of_classes
        self.config = config


    # ------------------------------------------------------
    # From config passed to method constructs CNN (Darknet53/Yolov3),
    # uses CNNBlock, ResidualBlock and ScalePrediction from cnn_utils
    def _constructNeuralNetwork(self, config, in_channels):

        # print(self.in_channels)

        # in_channels = self.in_channels
        net = nn.ModuleList()
        for block in self.config:

            # Construction of CNNBlock
            # CNNBlock changes number of channels, thus update
            if isinstance(block, tuple):
                layer, in_channels = self._buildCNNLayer(in_channels, block)

            # Construction of ResidualBlock
            # ResidualBlock doesn't change number of channels (no update needed)
            elif isinstance(block, list):
                layer = self._buildResidualLayer(in_channels, block)

            # Construction of ScalePrediction block
            elif isinstance(block, str):
                layer = self._buildScalePredictionLayer(in_channels, self.num_of_classes)

            # Construction of Upsampling block
            elif isinstance(block, dict):
                layer, in_channels = self._buildUpsampleLayer(in_channels, block)

            net.append(layer)
            
        self.out_channels = in_channels

        return net


    # ------------------------------------------------------
    # CNNBlock changes number of channels, update:
    # in_channels (for next layer) = out_channels (of this layer)
    def _buildCNNLayer(self, in_channels, block):

        out_channels, kernel_size, stride = block
        layer = CNNBlock(
            in_channels, 
            out_channels,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=1 if kernel_size == 3 else 0
        )

        return layer, out_channels


    # ------------------------------------------------------
    # block_type == 'R' builds block with residual behaviour,
    # block_type == 'C' just stacks CNNBlock
    # Residual layer doesn't change number of channels
    def _buildResidualLayer(self, in_channels, block):

        block_type, num_of_repeats = block
        layer = ResidualBlock(
            num_of_repeats, 
            in_channels,
            residual=True if block_type == 'R' else False
        )

        return layer


    # ------------------------------------------------------
    # ScalePrediction layer doesn't change number of channels,
    # output from this layer doesn't influence the network
    def _buildScalePredictionLayer(self, in_channels, num_of_classes):

        return ScalePrediction(in_channels, self.num_of_classes)


    # ------------------------------------------------------
    # Scale prediction layer modifies number of channels,
    # because of tensor concatenation right after this layer 
    def _buildUpsampleLayer(self, in_channels, block):

        return nn.Upsample(scale_factor=block["U"]), in_channels * 3




if __name__ == '__main__':


    from cnn_block import CNNBlock
    from residual_block import ResidualBlock
    from scale_prediction_block import ScalePrediction

    b = CNNBuilder(3, 6)
    net = b._constructNeuralNetwork(yolo)

    for block in net:

        print(type(block))
