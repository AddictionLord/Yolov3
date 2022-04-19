import torch
import torch.nn as nn

from utils import intersectionOverUnion




class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10


    # ------------------------------------------------------
    # target shape: [num_of_anchors, cells_x, cells_y, bounding_box/anchor_data]
    def forward(self, predictions, target, anchors):

        Iobj = target[..., 0] == 1
        Inoobj = target[..., 0] == 0

        # loss when there is no object
        without_obj = self.bce()

        # loss when there is object
        with_obj = self.mse()

        


if __name__ == "__main__":

    pass


