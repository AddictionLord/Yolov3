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

    # # TODO: Find out why create PerfTensor doesnt work on any image
    # loss_fcn = Loss(testing=True)
    # loader = getValLoader([103], False)
    # # loader = getValLoader(loadbar=False)
    # for img, targets in loader:

    #     targets = TargetTensor.fromDataLoader(config.ANCHORS, targets)
    #     anchors = torch.tensor(config.ANCHORS, dtype=torch.float16, device=config.DEVICE)
    #     preds, _ = createPerfectPredictionTensor(targets.tensor, anchors)

    #     testLossFcn(targets, preds, anchors, loss_fcn)
    #     input()

    # bce = torch.nn.BCEWithLogitsLoss()
    # a = torch.rand(1, 2, 3, 3, 6)
    # print(bce(a, a))


#%%
    import sys
    sys.path.insert(1, '/home/s200640/thesis/src/')
    import torch 
    import config
    from loss import Loss

    target = torch.zeros(1, 3, 3, 3, 6)
    preds = torch.zeros(1, 3, 3, 3, 11)

    def inv_sig(x):
        return -torch.log((1 / x) - 1)

    anchors = torch.tensor(config.ANCHORS, dtype=torch.float16, device='cpu')

    target[0, 0, 1, 1, 0] = 1
    target[0, 0, 1, 1, 1:5] = 0.5
    target[0, 0, 1, 1, 5] = 0

    preds[..., 0] = -torch.inf
    preds[0, 0, 1, 1, 0:3] = inv_sig(target[0, 0, 1, 1, 0:3]).clip(max=1e+6)
    preds[0, 0, 1, 1, 3:5] = torch.log(target[0, 0, 1, 1, 3:5] / anchors[0, ...][0])
    preds[0, 0, 1, 1, 5:] = torch.nn.functional.one_hot(target[0, 0, 1, 1, 5].long(), 6)

    # test = torch.zeros_like(target)
    # test[0, 0, 1, 1, 0:3] = torch.sigmoid(preds[0, 0, 1, 1, 0:3])
    # test[0, 0, 1, 1, 3:5] = anchors[0, ...][0] * torch.exp(preds[0, 0, 1, 1, 3:5])
    # test[0, 0, 1, 1, 5] = torch.argmax(preds[0, 0, 1, 1, 5:], dim=-1)

    loss_fcn = Loss()
    loss = loss_fcn(preds, target, anchors[0, ...])
    print(loss)



# %%
