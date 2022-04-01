import torch
import torch.nn as nn
import numpy as np

from blocks.residual_block import ResidualBlock
from blocks.cnn_builder import CNNBuilder
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


class Darknet(nn.Module, CNNBuilder):
    def __init__(self, in_channels: int, pretrained=True, config=config):
        super(Darknet, self).__init__()

        self.in_channels = in_channels
        self.config = config
        self.weights_handler = None
        self.concatenation = list()
        self.darknet = self._constructNeuralNetwork(config, in_channels)
        self.loadPretrainedModel(darknet53_path) if pretrained else None


    # ------------------------------------------------------
    # Iterates over all layers, saves tensors after ResidualBlock
    # with 8 repeats to concatenate tensor later (route connections)
    @torch.no_grad()
    def forward(self, x: torch.tensor):

        for layer in self.darknet:

            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_of_repeats == 8:
                self.concatenation.append(x)

        return x


    # ------------------------------------------------------
    # Loading darknet53 weights from darknet53.conv.74 file
    def loadPretrainedModel(self, src: str):

        # class Darknet is only one who carries WeightsHandler, 
        # others just use and forget it immediatly after
        self.weights_handler = WeightsHandler(src)
        for layer in self.darknet:

            layer.loadWeights(self.weights_handler)

        print('Pretrained model (darknet53) weights loaded..')


    # ------------------------------------------------------
    # From darknet53 there are route connections after 
    # ResidualBlocks with 8 repeats, this method 
    def getTensorToConcatenate(self):

        return self.concatenation.pop()




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
def showDarknetBlocks():

    from blocks.cnn_block import CNNBlock
    from blocks.residual_block import ResidualBlock

    d = Darknet(3, config)
    num_of_blocks = len(d.darknet)
    num_of_layers = 0
    print(f'Number of blocks: {num_of_blocks}')

    for index, block in enumerate(d.darknet):

        print(type(block))




# ------------------------------------------------------
# Main - mostly for testing purposes
# ------------------------------------------------------
if __name__ == '__main__':

    showDarknetBlocks()
    testDarknetOutputSize()

    # t = torch.rand(1, 3, 255, 255, requires_grad=True)
    # d = Darknet(3)

    # out = d(t)
    # print(out.requires_grad)    



    