import torch
import torch.nn as nn

from darknet import Darknet
# from blocks.scale_prediction_block import ScalePrediction
from blocks.cnn_builder import (
    CNNBuilder,
    CNNBlock,
    ResidualBlock,
    ScalePrediction
)

from utils import WeightsHandler



'''
Darknet53 feature detector from PJ Redmon's Yolov3 object detection model
config file: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
architecture inspirations:
    (Aladdin Persson): https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/model.py#L152
    (Ayoosh Kathuria): https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/darknet.py#L435
    (Erik Lindernoren): https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/e54d52e6500b7bbd687769f7abfd8581e9f59fd1/pytorchyolo/models.py#L199

'''

yolo_config = [
    # Darknet-53 before yolo
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
    def __init__(self, yolo_config: list, in_channels=3, num_of_classes=6, pretrained=True):
        super(Yolov3, self).__init__()

        self.config = yolo_config
        self.in_channels = in_channels
        self.num_of_classes = num_of_classes

        self.darknet = Darknet(in_channels, pretrained=pretrained)
        # self.darknet.eval()
        self.yolo = self._constructNeuralNetwork(yolo_config, self.darknet.out_channels)
        self.yolo.apply(WeightsHandler.initWeights)
        # self.yolo.apply(Yolov3.focalWeights)


    # ------------------------------------------------------
    def forward(self, x: torch.tensor):

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

    
    # # ------------------------------------------------------
    # # Switches model to train mode except BatchNorm2D layers
    # def fineTuningTrainModel(self):

    #     self.yolo.train()
    #     for block in self.yolo:

    #         if isinstance(block, CNNBlock):
    #             pass

    #         elif isinstance(block, ResidualBlock):
    #             pass

    #         elif isinstance(block, ScalePrediction):
    #             pass


    # ------------------------------------------------------
    # Initialize weights for last layer to fit focalLoss
    @staticmethod
    def focalWeights(layer):

        if isinstance(layer, ScalePrediction):

            print(layer.block[1].block[0].bias)
            layer.block[1].block[0].bias.data.fill_(-2)
            print(layer.block[1].block[0].bias)


# ------------------------------------------------------
# Testing functions
# ------------------------------------------------------
def showYoloBlocks():
        
    from blocks.cnn_block import CNNBlock
    from blocks.residual_block import ResidualBlock

    y = Yolov3(yolo_config)
    num_of_blocks = len(y.yolo)
    print(f'Number of blocks: {num_of_blocks}')

    for index, block in enumerate(y.yolo):

        print(type(block))


# ------------------------------------------------------
def testYoloOutputSize():

    NUM_OF_CLASSES = 6
    model = Yolov3(yolo_config, 3, NUM_OF_CLASSES)

    BATCH_SIZE = 1
    INPUT_WIDTH = 256
    INPUT_HEIGHT = 256
    t = torch.rand(BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT)

    out = model(t)

    assert out[0].shape == (BATCH_SIZE, 3, INPUT_WIDTH // 32, INPUT_HEIGHT // 32, num_of_classes + 5)
    assert out[1].shape == (BATCH_SIZE, 3, INPUT_WIDTH // 16, INPUT_HEIGHT // 16, num_of_classes + 5)
    assert out[2].shape == (BATCH_SIZE, 3, INPUT_WIDTH // 8, INPUT_HEIGHT // 8, num_of_classes + 5)
    print('Test was successful - Image output size is correct!')

    print(f'Output shape of scale 0: {out[0].shape}')
    print(f'Output shape of scale 1: {out[1].shape}')
    print(f'Output shape of scale 2: {out[2].shape}')


# ------------------------------------------------------
def testEvalMode():

    y = Yolov3(yolo_config)
    print(f'Training: {y.darknet.training}')
    y.yolo.train()    
    print(f'Training after train: {y.darknet.training}')
    y.yolo.eval()    
    print(f'Training after eval: {y.darknet.training}')



# ------------------------------------------------------
# Main - mostly for testing purposes
# ------------------------------------------------------
if __name__ == "__main__":

    # showYoloBlocks()
    # testYoloOutputSize()
    # testEvalMode()

    y = Yolov3(yolo_config)
    # y.fineTuningTrainMode()

    t = torch.rand(1, 3, 255, 255, requires_grad=True)
    out = y(t)


