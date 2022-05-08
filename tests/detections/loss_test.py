import sys
sys.path.insert(1, '/home/s200640/thesis/src/')

import torch

import config
from yolo_trainer import YoloTrainer
from loss import Loss

from perfect_tensor import createPerfectPredictionTensor
from utils import (
    TargetTensor, 
    getValLoader, 
    convertDataToMAP,
)




# ------------------------------------------------------
def testLossFcn(targets, preds, anchors: torch.tensor, loss_fcn):

    for scale, (target, pred, anchor) in enumerate(zip(targets, preds, anchors)):

        # TODO: Make convertPredsToBB complete with classes too
        tensor, _ = TargetTensor.convertPredsToBoundingBox(pred.clone(), anchor)
        classes = torch.argmax(tensor[..., 5:], dim=-1).unsqueeze(-1)
        converted_pred = torch.cat((classes, tensor[..., 0:5]), dim=-1)

        converted_target = target[..., torch.Tensor([5,0,1,2,3,4]).long()]
        print(f'Tensor on scale {scale} are identical: {torch.allclose(converted_target, converted_pred)}')

    loss = targets.computeLossWith(preds, loss_fcn, debug=True)
    print(loss)



# ------------------------------------------------------
if __name__ == '__main__':

    # TODO: Find out why create PerfTensor doesnt work on any image
    loss_fcn = Loss(testing=True)
    # loader = getValLoader([103], False)
    loader = getValLoader(loadbar=False)
    for img, targets in loader:

        targets = TargetTensor.fromDataLoader(config.ANCHORS, targets)
        anchors = torch.tensor(config.ANCHORS, dtype=torch.float16, device=config.DEVICE)
        preds, _ = createPerfectPredictionTensor(targets.tensor, anchors)

        testLossFcn(targets, preds, anchors, loss_fcn)
        input()

    # bce = torch.nn.BCEWithLogitsLoss()
    # a = torch.rand(1, 2, 3, 3, 6)
    # print(bce(a, a))
