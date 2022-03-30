import torch
import torch.nn as nn
import numpy as np

from blocks.cnn_block import CNNBlock
from blocks.residual_block import ResidualBlock
from utils.weights_handler import WeightsHandler


'''
Darknet53 feature detector from PJ Redmon's Yolov3 object detection model
config file: https://github.com/pjreddie/darknet/blob/master/cfg/darknet53.cfg
pretrained model: https://pjreddie.com/media/files/darknet53.conv.74
architecture inspirations:
    (Aladdin Persson): https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/model.py#L152
    (Ayoosh Kathuria): https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/darknet.py#L435
    (Erik Lindernoren): https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/e54d52e6500b7bbd687769f7abfd8581e9f59fd1/pytorchyolo/models.py#L199

'''

darknet53_path = 'pretrained/darknet53.conv.74'

# TODO: Load darknet53 config from json/config file
config = [
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


class Darknet(nn.Module):
    def __init__(self, in_channels, pretrained=True, config=config):
        super(Darknet, self).__init__()

        self.in_channels = in_channels
        self.config = config
        self.weights_handler = None
        self.concatenation = list()
        self.darknet = self._constructDarknet53()
        self.loadPretrainedModel(darknet53_path) if pretrained else None


    # ------------------------------------------------------
    def forward(self, x):

        return self.darknet(x)


    # ------------------------------------------------------
    # From config stored in self.config constructs Darknet53,
    # uses CNNBlock and ResidualBlock imported from cnn_utils
    def _constructDarknet53(self):

        in_channels = self.in_channels
        darknet = nn.Sequential()
        for block in self.config:

            # Construction of CNNBlock and integration to darknet
            if isinstance(block, tuple):
                out_channels, kernel_size, stride = block
                cnn_block = nn.Sequential(
                    CNNBlock(
                        in_channels, 
                        out_channels,
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=1 if kernel_size == 3 else 0
                    )
                )

                # CNNBlock changes number of channels - update:
                in_channels = out_channels
                darknet = nn.Sequential(*darknet, *cnn_block)

            # Construction of ResidualBlock and integration to darknet
            # ResidualBlock doesn't change number of channels (no update needed)
            elif isinstance(block, list):
                block_type, num_of_repeats = block
                res_block = nn.Sequential(
                    ResidualBlock(
                        num_of_repeats, 
                        in_channels
                    )
                )

                darknet = nn.Sequential(*darknet, *res_block)

        return darknet


    # ------------------------------------------------------
    # Loading darknet53 weights from darknet53.conv.74 file
    def loadPretrainedModel(self, src):

        # class Darknet is only one who carries WeightsHandler, 
        # others just use and forget it immediatly after
        self.weights_handler = WeightsHandler(src)
        for layer in self.darknet:

            layer.loadWeights(self.weights_handler)

        print('Pretrained model (darknet53) weights loaded..')




# ------------------------------------------------------
# Testing functions
# ------------------------------------------------------
def testDarknetOutputSize():

    model = Darknet(3, config)

    BATCH_SIZE = 1
    INPUT_WIDTH = 256
    INPUT_HEIGHT = 256
    t = torch.rand(BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT)

    out = model(t)

    assert out.shape == (BATCH_SIZE, 1024, INPUT_WIDTH // 32, INPUT_HEIGHT // 32)
    print('Test was successful - Image output size is correct!')





# ------------------------------------------------------
# Main - mostly for testing purposes
# ------------------------------------------------------
if __name__ == '__main__':

    d = Darknet(3, config)
    num_of_blocks = len(d.darknet)
    num_of_layers = 0
    print(f'Number of blocks: {num_of_blocks}')

    for index, block in enumerate(d.darknet):

        print(type(block))
        if isinstance(block, ResidualBlock):
            print(len(block.block))
            num_of_layers += len(block.block)

        else:
            num_of_layers += 1


    print(num_of_layers)


    