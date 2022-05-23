import torch
import sys
sys.path.insert(1, '/home/s200640/thesis/src/')
import copy

import config 
from utils import getValLoader, TargetTensor

from torch.nn.functional import one_hot




# ------------------------------------------------------------
# From anchors and dataloader returns perfect tensor for testing purposes
def createPerfectPredictionTensor(loader, anchors):


    def inv_sig(x):
        return -torch.log((1 / x) - 1)

    if isinstance(loader, (list, TargetTensor)):
        targets = loader
        image = None

    else:
        image, targets = next(iter(loader))
        targets = [target.detach().clone().requires_grad_(False).to(config.DEVICE) for target in targets]

    preds = [torch.zeros_like(i, device=config.DEVICE) for i in targets]
    for scale, target in enumerate(targets):

        # plot_image(image[0].permute(1,2,0).detach().cpu())

        condition = (target[..., 0] == 1)
        # condition = condition.repeat(1, 1, 1, 1, 6)#.reshape(batch_size, -1, 6)
        # pred_condition = condition.repeat(1, 1, 1, 1, 11)
        # target[condition].reshape(-1, 6)
        # print(target[0][condition].reshape(-1, 6))

        # idx = (target[..., 0:1] == 1).nonzero().tolist()[0]
        # values_idx = values_idx[..., 2:4].tolist()

        batch = target.shape[0]
        num_anchors = target.shape[1]
        num_of_cells = target.shape[2]
        preds[scale] = torch.zeros(batch, num_anchors, num_of_cells, num_of_cells, 11).to(config.DEVICE)

        # preds[scale][..., 0:3] = inv_sig(target[..., 0:3]).to(config.DEVICE)
        # # preds[0][..., 3:5] = torch.log(1e-16 + target[0][..., 3:5] / torch.tensor(anchors[0]).reshape(1, 3, 1, 1, 2))
        # # preds[scale][..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors[scale, ...].reshape(1, 3, 1, 1, 2)).to(config.DEVICE)
        # preds[scale][..., 3:5] = torch.log(target[..., 3:5] / anchors[scale, ...].reshape(1, 3, 1, 1, 2)).to(config.DEVICE)

        # classes = target[..., -1].long().to(config.DEVICE)
        # preds[scale][..., 5:] = one_hot(classes, 6)
        # preds[scale][..., 5:] = torch.where(preds[scale][..., 5:] == 1.0, torch.inf, 0.0)

        # #--------------------------------------------------------
        # back_target = torch.zeros(target.shape).to(config.DEVICE)
        # # back_target = preds[scale].clone()
        # back_target[..., 0:3] = torch.sigmoid(preds[scale][..., 0:3])
        # # back_target[0][..., 3:5] = torch.exp(preds[0][..., 3:5]) * torch.tensor(anchors[0]).reshape(1, 3, 1, 1, 2)
        # back_target[..., 3:5] = torch.exp(preds[scale][..., 3:5]) * anchors[scale, ...].reshape(1, 3, 1, 1, 2)
        # back_target[..., 5] = torch.argmax(preds[scale][..., 5:], dim=-1)

        # Change values only on positions with number (where condition is true)
        preds[scale][..., 0:3][condition] = inv_sig(target[..., 0:3][condition]).to(config.DEVICE)
        # preds[0][..., 3:5] = torch.log(1e-16 + target[0][..., 3:5] / torch.tensor(anchors[0]).reshape(1, 3, 1, 1, 2))
        # preds[scale][..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors[scale, ...].reshape(1, 3, 1, 1, 2)).to(config.DEVICE)
        preds[scale][..., 3:5][condition] = torch.log(target[..., 3:5] / anchors[scale, ...].reshape(1, 3, 1, 1, 2))[condition].to(config.DEVICE)

        classes = target[..., -1][condition].long().to(config.DEVICE)
        preds[scale][..., 5:][condition] = one_hot(classes, 6).float()
        preds[scale][..., 5:][condition] = torch.where(preds[scale][..., 5:][condition] == 1.0, torch.inf, 0.0)

        #--------------------------------------------------------
        back_target = torch.zeros(target.shape).to(config.DEVICE)
        # back_target = preds[scale].clone()
        back_target[..., 0:3][condition] = torch.sigmoid(preds[scale][..., 0:3][condition])
        # back_target[0][..., 3:5] = torch.exp(preds[0][..., 3:5]) * torch.tensor(anchors[0]).reshape(1, 3, 1, 1, 2)
        back_target[..., 3:5][condition] = (torch.exp(preds[scale][..., 3:5]) * anchors[scale, ...].reshape(1, 3, 1, 1, 2))[condition]
        back_target[..., 5][condition] = torch.argmax(preds[scale][..., 5:][condition], dim=-1).float()

        print(f'tensors are same: {torch.allclose(target, back_target)}')

    return preds, image




# ------------------------------------------------------------
if __name__ == '__main__':

    # CUDA_LAUNCH_BLOCKING=1
    # config.DEVICE = 'cpu'

    anchors = torch.tensor(config.ANCHORS, dtype=torch.float16, device=config.DEVICE)
    # loader = getValLoader([103], False)
    loader = getValLoader()
    createPerfectPredictionTensor(loader, anchors)

    # ---------------------------------
    # Second option, same    
    loader = getValLoader([103], False)
    for img, targets in loader:

        targets = TargetTensor.fromDataLoader(config.ANCHORS, targets)
        createPerfectPredictionTensor(targets.tensor, anchors)



    # preds, image = createPerfectPredictionTensor(val_loader, anchors)
    # plotDetections(model, image.to(device), config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, anchors, preds)


#%%
import torch

t = torch.zeros(1, 3, 3, 3, 2)
p = torch.zeros(1, 3, 3, 3, 5)

t[0, 0, 1, 1, 0] = 1
t[0, 0, 1, 2, 0] = 1
t[0, 0, 1, 2, 1] = 2
t[0, 0, 1, 1, 1] = 4

condition = t[..., 0] == 1
p[..., 0:1][condition] = t[..., 0:1][condition] * 2
p[..., 1:2][condition] = t[..., 1:2][condition] * 4
p[..., 0:2][condition] = t[..., 0:2][condition] * 4
print(t[condition])
print(p[condition])

# %%
