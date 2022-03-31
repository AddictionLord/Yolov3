import torch
import torch.nn as nn

from darknet import Darknet
from blocks.scale_prediction_block import ScalePrediction
from blocks.cnn_builder import CNNBuilder



'''
Darknet53 feature detector from PJ Redmon's Yolov3 object detection model
config file: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
architecture inspirations:
    (Aladdin Persson): https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/model.py#L152
    (Ayoosh Kathuria): https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/darknet.py#L435
    (Erik Lindernoren): https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/e54d52e6500b7bbd687769f7abfd8581e9f59fd1/pytorchyolo/models.py#L199

'''

config = [
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


class Yolov3(nn.Module, CNNBuilder):
    def __init__(self, config, in_channels=3, num_of_classes=6):
        super(Yolov3, self).__init__()

        self.config = config
        self.in_channels = in_channels
        self.num_of_classes = num_of_classes

        self.darknet = Darknet(in_channels, pretrained=True)
        self.yolo = self._constructNeuralNetwork(config, self.darknet.out_channels)


    # ------------------------------------------------------
    def forward(self, x):

        outputs = list()
        x = self.darknet(x)
        for layer in self.yolo:

            # Scale prediction outputs prediction tensor, but don't 
            # influence next layers, thus continuation after
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            # After upsampling layer output is concatenated with
            # tensor from darknet residual layer - route connection
            if isinstance(layer, nn.Upsample):
                x = torch.cat((x, self.darknet.getTensorToConcatenate()), dim=1)

        return outputs



# ------------------------------------------------------
# Testing functions
# ------------------------------------------------------
def showYoloBlocks():
        
    from blocks.cnn_block import CNNBlock
    from blocks.residual_block import ResidualBlock

    y = Yolov3(config)
    num_of_blocks = len(y.yolo)
    print(f'Number of blocks: {num_of_blocks}')

    for index, block in enumerate(y.yolo):

        print(type(block))


def testYoloOutputSize():

    num_of_classes = 6
    model = Yolov3(config, 3, num_of_classes)

    BATCH_SIZE = 1
    INPUT_WIDTH = 256
    INPUT_HEIGHT = 256
    t = torch.rand(BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT)

    out = model(t)

    assert out[0].shape == (BATCH_SIZE, 3, INPUT_WIDTH // 32, INPUT_HEIGHT // 32, num_of_classes + 5)
    assert out[1].shape == (BATCH_SIZE, 3, INPUT_WIDTH // 16, INPUT_HEIGHT // 16, num_of_classes + 5)
    assert out[2].shape == (BATCH_SIZE, 3, INPUT_WIDTH // 8, INPUT_HEIGHT // 8, num_of_classes + 5)
    print('Test was successful - Image output size is correct!')



# ------------------------------------------------------
# Main - mostly for testing purposes
# ------------------------------------------------------
if __name__ == "__main__":

    # showYoloBlocks()
    testYoloOutputSize()


