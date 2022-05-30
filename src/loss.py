import torch
import torch.nn as nn

import config
from utils import intersectionOverUnion, TargetTensor


'''
https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff
https://towardsdatascience.com/yolo-v3-explained-ff5b850390f

'''


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Values from Yolov1 paper
        self.lambda_coord = config.LAMBDA_COORD #10
        self.lambda_noobj = config.LAMBDA_NOOBJ #0.5 #10


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

        noobj_loss = self.bce(predictions[..., 0:1][Inoobj], target[..., 0:1][Inoobj])

        # loss when there is object
        preds, anchors = TargetTensor.convertPredsToBoundingBox(predictions, anchors.clone())
        ious = intersectionOverUnion(preds[..., 1:5][Iobj], target[..., 1:5][Iobj])
        # obj_loss = self.bce(preds[..., 0:1][Iobj], ious * target[..., 0:1][Iobj])
        print('preds: ', predictions[..., 0:1][Iobj])
        print('targets: ', target[..., 0:1][Iobj])
        obj_loss = self.bce(predictions[..., 0:1][Iobj], target[..., 0:1][Iobj])
        print()
        print(f'Object loss: {obj_loss}')

        # loss when there is no object
        # noobj_loss = self.bce(preds[..., 0:1][Inoobj], target[..., 0:1][Inoobj])
        print(f'No object loss: {noobj_loss}')

        # box coordinates loss
        # xy_loss = self.mse(preds[..., 1:3][Iobj], target[..., 1:3][Iobj])
        # target_wh_recomputed = torch.log(1e-8 + target[..., 3:5] / anchors)
        # wh_loss = self.mse(predictions[..., 3:5][Iobj], target_wh_recomputed[Iobj])
        # box_loss = torch.mean(torch.tensor([xy_loss, wh_loss]))

        # xy_loss = self.mse(preds[..., 1:3][Iobj], target[..., 1:3][Iobj])
        # wh_loss = self.mse(preds[..., 3:5][Iobj], target[..., 3:5][Iobj])
        # box_loss = torch.mean(torch.tensor([xy_loss, wh_loss]))
        box_loss = self.mse(preds[..., 1:5][Iobj], target[..., 1:5][Iobj])
        print(f'Box loss: {box_loss}')

        # class loss
        class_loss = self.entropy(predictions[..., 5:][Iobj], target[..., 5][Iobj].long())
        print(f'Class loss: {class_loss}')
        
        # Convert nan values to 0, torch.nan_to_num not available in dev torhc version
        noobj_loss[torch.isnan(noobj_loss)]     = 0
        # obj_loss[torch.isnan(obj_loss)]         = 0
        box_loss[torch.isnan(box_loss)]         = 0
        class_loss[torch.isnan(class_loss)]     = 0

        # loss fcn
        return (self.lambda_coord * box_loss 
            + obj_loss * 10
            + self.lambda_noobj * noobj_loss 
            + class_loss
        )



# ------------------------------------------------------
def getOptimalTargetAndPreds():

    # anchors = torch.rand(3, 2)
    anchors = torch.tensor([[1, 1], [2, 2], [3, 3]])
    anchors_reshaped = anchors.reshape(1, len(anchors), 1, 1, 2)
    predictions = torch.zeros(1, 3, 13, 13, 11)
    target = predictions[..., 0:6].detach().clone() #torch.rand(1, 3, 13, 13, 6)

    # this is for score
    predictions[0, 0, 1, 1, 0] = 7 # 7 because sigmoid(7) = 0.99
    target[0, 0, 1, 1, 0] = 1

    # this is for bbox
    predictions[0, 0, 1, 1, 1:5] = torch.rand(4)
    target[..., 1:3] = torch.sigmoid(predictions[..., 1:3])
    target[..., 3:5] = torch.exp(predictions[..., 3:5]) * anchors_reshaped

    # this is for class
    predictions[0, 0, 1, 1, 5:] = torch.tensor([0, 1, 0, 0, 0, 0])
    target[0, 0, 1, 1, 5] = 1

    # print(predictions[0, 0, 1, 1, ...])
    # print(target[0, 0, 1, 1, ...])
    # print(target[0, 0, :, :, 5])

    return target, predictions, anchors



# ------------------------------------------------------
if __name__ == "__main__":


    # def sig(x):
    #     return 1 / (1 + torch.exp(torch.tensor(-x)))

    # def inv_sig(x):
    #     return -torch.log((1 / x) - 1)

    # num = torch.tensor([0.5, 0.2, 0.1, -0.2])
    # num1 = inv_sig(sig(num))
    # num2 = inv_sig(torch.sigmoid(num))
    # print(num1)
    # print(num2)
    # target, predictions, anchors = getOptimalTargetAndPreds()

    # l = Loss()
    # loss = l(predictions.detach().clone(), target.detach().clone(), anchors.detach().clone())
    # print(f"Loss: {loss}")

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

    bce = nn.BCEWithLogitsLoss()
    i = torch.FloatTensor([7, 7])
    t = torch.FloatTensor([1, 1])
    print(bce(i, t))
