import torch
import torch.nn as nn

from utils import intersectionOverUnion, TargetTensor


'''
https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff
https://towardsdatascience.com/yolo-v3-explained-ff5b850390f

'''


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Values from Yolov1 paper
        # self.lambda_coord = 5 #10
        # self.lambda_noobj = 0.5 #10

        self.lambda_coord = 10
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

        # loss when there is no object
        noobj_loss = self.bce(predictions[..., 0:1][Inoobj], target[..., 0:1][Inoobj])
        noobj_loss[torch.isnan(noobj_loss)] = 0

        # loss when there is object
        preds, anchors = TargetTensor.convertPredsToBoundingBox(predictions.detach().clone(), anchors)
        ious = intersectionOverUnion(preds[..., 1:5][Iobj], target[..., 1:5][Iobj]).detach()
        obj_loss = self.bce(preds[..., 0:1][Iobj], ious * target[..., 0:1][Iobj])

        # box coordinates loss
        xy_loss = self.mse(preds[..., 1:3][Iobj], target[..., 1:3][Iobj])
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors)
        wh_loss = self.mse(predictions[..., 3:5][Iobj], target[..., 3:5][Iobj])
        box_loss = torch.mean(torch.tensor([xy_loss, wh_loss]))

        # class loss
        class_loss = self.entropy(predictions[..., 5:][Iobj], target[..., 5:6][Iobj].long().squeeze())

        # loss fcn
        loss = (self.lambda_coord * box_loss
            + obj_loss 
            + self.lambda_noobj * noobj_loss
            + class_loss
        )

        return loss




# ------------------------------------------------------
if __name__ == "__main__":


    predictions = torch.rand(1, 3, 13, 13, 11)
    target = torch.rand(1, 3, 13, 13, 6)
    target[..., 0:1] = 1
    anchors = torch.rand(3, 2)

    l = Loss()
    loss = l(predictions.detach().clone(), target.detach().clone(), anchors.detach().clone())
    print(f"Loss: {loss}")

    # -------------------------------------------
    # mse = nn.MSELoss()
    # cnt = 0
    # for _ in range(100):

    #     i = torch.rand(2, 4)
    #     t = torch.rand(2, 4)

    #     def my():

    #         a = mse(i[..., 0:2], t[..., 0:2])
    #         b = mse(i[..., 2:4], t[..., 2:4])

    #         return torch.mean(torch.tensor([a, b]))
    #         # return (a + b) / 2 


    #     mymse = my()
    #     tmse = mse(i, t)

    #     cnt += torch.allclose(mymse, tmse)

    # print(cnt)
