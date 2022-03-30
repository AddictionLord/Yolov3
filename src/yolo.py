import torch
import torch.nn as nn

from darknet import Darknet
from blocks.cnn_block import CNNBlock

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


config = [
    (512, 1, 1),
    (1024, 3, 1),
    ["R", 1],
    (512, 1, 1),
    "S",

    (256, 1, 1),
    "U",

    (256, 1, 1),
    (512, 3, 1),
    ["R", 1],
    (256, 1, 1),
    "S",

    (128, 1, 1),
    "U",

    (128, 1, 1),
    (256, 3, 1),
    ["R", 1],
    (128, 1, 1),
    "S"
]


class Yolov3(nn.Module):
    def __init__(self, config, in_channels=3, num_of_classes=6):
        super(Yolov3, self).__init__()

        self.config = config
        self.in_channels = in_channels
        self.num_of_classes = num_of_classes

        self.darknet = Darknet(in_channels, pretrained=True)
        self.yolo = self._constructYolov3(config)


    # ------------------------------------------------------
    def forward(self, x):

        x = self.darknet(x)
        for layer in self.yolo:

            x = layer(x)

        return x


    # ------------------------------------------------------
    def _constructYolov3(self):

        yolo = nn.ModuleList()
        for block in self.config:

            # Construction of CNNBlock and integration to darknet
            if isinstance(block, tuple):
                out_channels, kernel_size, stride = block
                layer = CNNBlock(
                    in_channels, 
                    out_channels,
                    kernel_size=kernel_size, 
                    stride=stride, 
                    padding=1 if kernel_size == 3 else 0
                )

                # CNNBlock changes number of channels - update:
                in_channels = out_channels

                





if __name__ == "__main__":

    y = Yolov3(config)