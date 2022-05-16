import torch
from torch import nn
# from nn.functional.F import softmax, one_hot




# --------------------------------------------------------------
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()

        loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.loss_fcn = loss_fcn
        self.reduction = reduction

        self.gamma = gamma
        self.alpha = alpha


    # --------------------------------------------------------------
    def forward(self, input, target):
        loss = self.loss_fcn(input, target.float())
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss