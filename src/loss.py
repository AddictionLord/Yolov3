import torch
import torch.nn as nn

from utils import intersectionOverUnion, TargetTensor


'''
https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff

'''


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_coord = 10 #5
        self.lambda_obj = 1
        self.lambda_noobj = 10


    # ------------------------------------------------------
    # target shape: [batches, num_of_anchors, cells_x, cells_y, bounding_box/anchor_data]
    #               bounding_box/anchor_data: [score, x, y, w, h, classification]
    # prediction shape: [batches, num_of_anchors, cells_x, cells_y, 5 + num_of_classes (= 11)]
    #                   5 + num_of_classes (= 11): [score, x, y, w, h, num_of_classes..(6)]
    # anchors shape: [3, 2] - for each scale we have 3 anchors with 2 values
    # This computes loss only for one scale (need to call 3 times)
    def forward(self, predictions, target, anchors):

        Iobj = target[..., 0] == 1
        Inoobj = target[..., 0] == 0

        # box coordinates loss
        

        # loss when there is no object
        noobj_loss = self.bce(predictions[..., 0:1][Inoobj], target[..., 0:1][Inoobj])

        # loss when there is object
        preds, anchors = TargetTensor.convertPredsToBoundingBox(predictions, anchors)
        ious = intersectionOverUnion(preds[..., 1:5], target[..., 1:5]).detach()
        obj_loss = self.bce(preds[..., 0:1][Iobj], target[..., 0:1][Iobj] * ious)

        


# ------------------------------------------------------
if __name__ == "__main__":

    bce = nn.BCEWithLogitsLoss()

    t = torch.tensor([1., 0., 1., 0., 0., 0.]).reshape(6, 1).repeat(1, 6)
    o = torch.tensor([1., 0., 1., 1., 0., 0.]).reshape(6, 1).repeat(1, 6)

    print(o[..., 0:1][True])

    loss = bce((o[..., 0:1][True]), (t[..., 0:1][True]))
    print(loss)

    # l = Loss()

    # predictions = torch.rand(1, 3, 13, 13, 11)
    # target = torch.rand(1, 3, 13, 13, 5)
    # l(predictions, target, None)