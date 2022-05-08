import torch
import torch.nn as nn

import config
from utils import intersectionOverUnion, TargetTensor, iou


'''
https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff
https://towardsdatascience.com/yolo-v3-explained-ff5b850390f
https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e

'''


class Loss(nn.Module):
    def __init__(self, testing=False):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Values from Yolov1 paper
        self.lambda_box = config.LAMBDA_COORD #10
        self.lambda_noobj = config.LAMBDA_NOOBJ #0.5 #10
        self.lambda_obj = config.LAMBDA_OBJ
        self.lambda_class = config.LAMBDA_CLASS

        def returnsOnes(pred, targets):

            return torch.ones(pred.shape[0], device=config.DEVICE).reshape(-1, 1)

        self.iou_fcn = returnsOnes if testing else iou




    # ------------------------------------------------------
    # target shape: [batches, num_of_anchors, cells_x, cells_y, bounding_box/anchor_data]
    #               bounding_box/anchor_data: [score, x, y, w, h, classification]
    # prediction shape: [batches, num_of_anchors, cells_x, cells_y, 5 + num_of_classes (= 11)]
    #                   5 + num_of_classes (= 11): [score, x, y, w, h, num_of_classes..(6)]
    # anchors shape: [3, 2] - for each scale we have 3 anchors with 2 values
    # This computes loss only for one scale (need to call 3 times)
    def forward(self, predictions, target, anchors, debug=False):

        # ------------------------------------------------------
        Iobj = target[..., 0] == 1
        Inoobj = target[..., 0] == 0
        # d = torch.ones(target[..., 0].shape).to(config.DEVICE) - Iobj.float()


        # ------------------------------------------------------
        # BCE uses sigmoid inside!!!
        noobj_loss = self.bce(predictions[..., 0:1][Inoobj].clip(min=-1e+16), target[..., 0:1][Inoobj])
        # print(torch.allclose(predictions[..., 0:1][Inoobj].clip(min=-1e+16), target[..., 0:1][Inoobj]))


        # ------------------------------------------------------
        # loss when there is object
        preds, anchors = TargetTensor.convertPredsToBoundingBox(predictions, anchors.clone())
        ious = self.iou_fcn(preds[..., 1:5][Iobj], target[..., 1:5][Iobj])
        obj_loss = self.bce(predictions[..., 0:1][Iobj].clip(max=1e+16), ious * target[..., 0:1][Iobj])

        # preds, anchors = TargetTensor.convertPredsToBoundingBox(predictions, anchors.clone())
        # ious = iou(preds[..., 1:5][Iobj], target[..., 1:5][Iobj])
        # obj_loss = self.bce(preds[..., 0:1][Iobj], ious * target[..., 0:1][Iobj])

        # BAD - WE HAVE NOOBJ LOSS FOR THAT
        # New approach, objectness computed for all cells, even without bb
        # preds, anchors = TargetTensor.convertPredsToBoundingBox(predictions, anchors.clone())
        # ious = intersectionOverUnion(preds[..., 1:5].reshape(-1, 4), target[..., 1:5].reshape(-1, 4))
        # obj_loss = self.bce(preds[..., 0:1].reshape(-1, 1), ious * target[..., 0:1].reshape(-1, 1))
        # obj_loss = self.bce(preds[..., 0:1].reshape(-1, 1), target[..., 0:1].reshape(-1, 1))

        # obj_loss = self.bce(predictions[..., 0:1][Iobj], target[..., 0:1][Iobj])
        # print()


        # ------------------------------------------------------
        # box coordinates loss
        # xy_loss = self.mse(preds[..., 1:3][Iobj], target[..., 1:3][Iobj])
        # target_wh_recomputed = torch.log(1e-8 + target[..., 3:5] / anchors)
        # wh_loss = self.mse(predictions[..., 3:5][Iobj], target_wh_recomputed[Iobj])
        # box_loss = torch.mean(torch.tensor([xy_loss, wh_loss]))

        # xy_loss = self.mse(preds[..., 1:3][Iobj], target[..., 1:3][Iobj])
        # wh_loss = self.mse(preds[..., 3:5][Iobj], target[..., 3:5][Iobj])
        # box_loss = torch.mean(torch.tensor([xy_loss, wh_loss]))
        box_loss = self.mse(preds[..., 1:5][Iobj], target[..., 1:5][Iobj])


        # ------------------------------------------------------
        # class loss
        class_loss = self.entropy(predictions[..., 5:][Iobj].clip(max=1e+16), target[..., 5][Iobj].long())
        
        # Convert nan values to 0, torch.nan_to_num not available in dev torhc version
        # noobj_loss[torch.isnan(noobj_loss)]     = 0
        # obj_loss[torch.isnan(obj_loss)]         = 0
        # box_loss[torch.isnan(box_loss)]         = 0
        # class_loss[torch.isnan(class_loss)]     = 0

        if config.DEBUG or debug:
            print('\npreds:\n', preds[..., 0:1][Iobj])
            print('targets:\n', target[..., 0:1][Iobj])

            print(f'Object loss: {obj_loss}')
            print(f'No object loss: {noobj_loss}')
            print(f'Class loss: {class_loss}')
            print(f'Box loss: {box_loss}')

        # loss fcn
        return (self.lambda_box * box_loss 
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_class * class_loss
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
    i = torch.FloatTensor([torch.inf, torch.inf]).clip(min=1e-16, max=1e+16)
    t = torch.FloatTensor([1, 1])
    print(bce(i, t))

    def inv_sig(x):
        return -torch.log((1 / x) - 1)

    print(inv_sig(torch.tensor(1)))
    print(inv_sig(torch.tensor(0)))
