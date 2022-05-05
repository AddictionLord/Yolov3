import sys
import torch
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(1, '/home/s200640/thesis/src/')
import config






# ------------------------------------------------------
# 1. Monitoring of training process, store data
# 2. Scheduling of learning rate
# 3. Saving checkpoint of model at best mAP score, saving dataFrame
class TrainSupervisor:
    def __init__(self, device, optimizer, epoch=0):

        self.last_epoch = epoch
        self.data = pd.DataFrame(columns=config.COLS)
        self.mAP = MeanAveragePrecision(box_format='cxcywh').to(device)
        self.scheduler = ReduceLROnPlateau(
            optimizer, 
            factor=0.5, 
            patience=5, 
            min_lr=1e-7, 
            verbose=True, 
            threshold=1e-4,
            cooldown=2
        )


    # ------------------------------------------------------
    # Accepts targets and pred from whole batch, computes new mAP values
    # and update dataFrame
    def update(preds_bboxes, target_bboxes, loss, val_loss):

        for preds, targets in zip(preds_bboxes, target_bboxes):

            preds, targets = TrainSupervisor._convertDataToMAP(preds, targets)
            self.mAP.update(preds, targets)

        mAP = self.mAP.compute()
        self.mAP.reset()
        self.updateDataFrame(mAP, loss, val_loss)
        

    # ------------------------------------------------------
    # Creates new index in dataFrame with passed values
    def updateDataFrame(self, mAP, loss, val_loss):

        lrate = self.scheduler.optimizer.param_groups[0]['lr'] if hasattr(self, 'scheduler') else config.LEARNING_RATE

        # Creating pandas series to integrate into DataFrame 
        e = pd.Series([self.last_epoch], name='epoch', dtype=np.int16)
        l = pd.Series([loss], name='loss', dtype=np.float16)
        vl = pd.Series([val_loss], name='val_loss', dtype=np.float16)
        lr = pd.Series([lrate], name='learning_rate', dtype=np.float16)

        row = pd.Series(mAP, dtype=np.float16)
        row = row[row >= 0].to_frame().T
        row = pd.concat((e, lr, l, vl, row), axis=1, ignore_index=True)
        row.columns = config.COLS

        self.data = pd.concat((self.data, row), axis=0, ignore_index=True)


    # ------------------------------------------------------
    # Metrics used to evaluate object detection models
    # Accepts tensors with shape[-1] = 6: [class, score, x, y, w, h]
    @staticmethod
    def _convertDataToMAP(preds: torch.tensor, targets: torch.tensor):

        preds = [
            dict(
                boxes=preds[..., 2:6].to(torch.float32),
                scores=preds[..., 1].to(torch.float32),
                labels=preds[..., 0].to(torch.int32),
            )
        ]
        targets = [
            dict(
                boxes=targets[..., 2:6].to(torch.float32),
                labels=targets[..., 0].to(torch.int32),
            )
        ]

        return preds, targets

        


# ------------------------------------------------------
if __name__ == '__main__':

    preds = [
        dict(
            boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0], [258.0, 241.0, 606.0, 285.0]]),
            scores=torch.tensor([0.536, 0.536]),
            labels=torch.tensor([0, 0]),
        )
    ]
    target = [
        dict(
            boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0], [214.0, 241.0, 562.0, 285.0]]),
            labels=torch.tensor([0, 0]),
        )
    ]

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    # mAP = metric.compute()

    from yolo import Yolov3
    model = Yolov3(config.yolo_config)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )


    # t = TrainSupervisor(config.DEVICE, optimizer=optimizer)
    # t._updateDataFrame(mAP, 0, 2)
    # t._updateDataFrame(mAP, 3, 12)
    pprint(metric)



    # df = pd.DataFrame(columns=config.COLS)

    # e = pd.Series([20], name='epoch', dtype=np.float16)
    # l = pd.Series([2], name='loss', dtype=np.float16)
    # vl = pd.Series([0], name='val_loss', dtype=np.float16)
    # lr = pd.Series([0.2], name='learning_rate', dtype=np.float16)

    # row = pd.Series(mAP, dtype=np.float16)
    # row = row[row >= 0].to_frame().T
    # row = pd.concat((e, lr, l, vl, row), axis=1, ignore_index=True)
    # row.columns = config.COLS
    # # print(s)

    # df = pd.concat((df, row), axis=0, ignore_index=True)
    # df = pd.concat((df, row), axis=0, ignore_index=True)

    # print(df)



