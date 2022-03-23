import torch
import torch.nn as nn
from cnn_utils import CNNBlock, ResidualBlock


'''
Darknet53 feature detector from Yolov3 object detection model
config file: https://github.com/pjreddie/darknet/blob/master/cfg/darknet53.cfg

'''


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
    def __init__(self, in_channels, config):
        super(Darknet, self).__init__()

        self.in_channels = in_channels
        self.config = config
        self.darknet = self._constructDarknet53()


    # ------------------------------------------------------
    def forward(self, x):

        return self.darknet(x)


    # ------------------------------------------------------
    def _constructDarknet53(self):

        in_channels = self.in_channels
        darknet = nn.Sequential()
        for block in self.config:

            if isinstance(block, tuple):
                out_channels, kernel_size, stride = block
                padding = 1 if kernel_size == 3 else 0
                cnn_block = nn.Sequential(
                    CNNBlock(in_channels, out_channels, kernel_size, stride, padding=padding)
                )

                in_channels = out_channels
                darknet = nn.Sequential(*darknet, *cnn_block)

            if isinstance(block, list):
                block_type, num_of_repeats = block
                res_block = nn.Sequential(
                    ResidualBlock(num_of_repeats, in_channels)
                )

                darknet = nn.Sequential(*darknet, *res_block)

        return darknet










if __name__ == '__main__':

    model = Darknet(3, config)
    # print(model.darknet)

    t = torch.rand(1, 3, 256, 256)
    out = model(t)

    print(out.shape)