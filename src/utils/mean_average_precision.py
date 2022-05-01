import torch
import sys

from torchmetrics.detection.mean_ap import MeanAveragePrecision
sys.path.insert(1, '/home/s200640/thesis/src/')


r'''
https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
'''


# ------------------------------------------------------
# Metrics used to evaluate object detection models
def convertDataToMAP(preds: list, target: list):

    preds = [
        dict(
            boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
            scores=torch.tensor([0.536]),
            labels=torch.tensor([0]),
        )
    ]
    target = [
        dict(
            boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
            labels=torch.tensor([0]),
        )
    ]





if __name__ == '__main__':

    preds = [
        dict(
            boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
            scores=torch.tensor([0.536]),
            labels=torch.tensor([0]),
        )
    ]
    target = [
        dict(
            boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
            labels=torch.tensor([0]),
        )
    ]

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    from pprint import pprint
    pprint(metric.compute())
