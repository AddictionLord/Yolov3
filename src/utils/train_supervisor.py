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
    def __init__(self, device, optimizer, epoch=0, name: str=None):

        self.name = name
        self.last_epoch = epoch
        self.val_loss = np.nan
        self.best_mAP = - np.inf
        self.mAP_dict = list()
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
    def state_dict(self, filename: str):

        state = {
            'filename': filename,
            'last_epoch': self.last_epoch,
            'val_loss': self.val_loss,
            'best_mAP': self.best_mAP,
            'mAP_dict': self.mAP_dict,
            'mAP': self.mAP.state_dict(),
        }
        path = f'./models/train_data/{filename}.pkl'
        self.data.to_pickle(path)
        print(f'[TRAIN SUPERVISOR]: Train data stored to {path}')
        
        return state


    # ------------------------------------------------------
    def load_state_dict(self, state_dict: dict):

        path = f'./models/train_data/{state_dict["filename"]}.pkl'
        self.name = state_dict["filename"]
        self.last_epoch = state_dict['last_epoch']
        self.val_loss = state_dict['val_loss']
        self.best_mAP = state_dict['best_mAP']
        self.mAP_dict = state_dict['mAP_dict']

        self.data = pd.read_pickle(path)
        print(self.data)
        print(f'[TRAIN SUPERVISOR]: Train data loaded from {path}')


    # ------------------------------------------------------
    # Accepts targets and pred from whole batch, updates mAP state
    def updateMAP(self, preds_bboxes, target_bboxes):

        for preds, targets in zip(preds_bboxes, target_bboxes):

            preds, targets = TrainSupervisor._convertDataToMAP(preds, targets)
            self.mAP.update(preds, targets)


    # ------------------------------------------------------
    # Accepts loss and val_loss, computes new mAP values and update dataFrame
    def update(self, loss):

        self.last_epoch += 1

        if self.mAP._update_called:
            self.mAP_dict = self.mAP.compute()
            self.mAP.reset()
            self.scheduler.step(self.mAP_dict["map"])
            self.mAP._update_called = False

        mAP = self.mAP_dict if isinstance(self.mAP_dict, dict) else [np.nan for _ in range(8)]
        self.updateDataFrame(mAP, loss, self.val_loss)

        return mAP["map"] if isinstance(mAP, dict) else - np.inf
        

    # ------------------------------------------------------
    # Creates new index in dataFrame with passed values
    def updateDataFrame(self, mAP, loss, val_loss):

        lrate = self.scheduler.optimizer.param_groups[0]['lr']

        # Creating pandas series to integrate into DataFrame 
        e = pd.Series([self.last_epoch], name='epoch', dtype=np.int32)
        l = pd.Series([loss], name='loss', dtype=np.float16)
        vl = pd.Series([val_loss], name='val_loss', dtype=np.float16)
        lr = pd.Series([lrate], name='learning_rate', dtype=np.float16)

        mAP = {k: v.cpu() for k, v in mAP.items()} if isinstance(mAP, dict) else mAP
        row = pd.Series(mAP, dtype=np.float16)
        row = row[row != -1].to_frame().T
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
    mAP = metric.compute()

    from yolo import Yolov3
    model = Yolov3(config.yolo_config)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )


    t = TrainSupervisor(config.DEVICE, optimizer=optimizer)

    t.updateDataFrame(mAP, 0, 2)
    t.updateDataFrame(mAP, 3, 12)

    t.updateDataFrame([np.nan for _ in range(8)], 0, 2)
    t.updateDataFrame([np.nan for _ in range(8)], 3, 12)

    print(t.data[['epoch', 'loss', 'val_loss', 'map', 'map_50', 'map_75']].tail(1))




    # e = pd.Series([2], name='epoch', dtype=np.int8)
    # l = pd.Series([20], name='loss', dtype=np.float16)
    # vl = pd.Series([22], name='val_loss', dtype=np.float16)
    # lr = pd.Series([0.001], name='learning_rate', dtype=np.float16)

    # # nan = pd.Series([np.nan for _ in range(8)], dtype=np.float16)
    # nan = pd.Series([np.nan for _ in range(8)], dtype=np.float16)
    # # nan = pd.Index(mAP.values(), dtype=np.float16)
    # nan = nan[nan != -1].to_frame().T

    # nan = pd.concat((e, lr, l, vl, nan), axis=1, ignore_index=True)

    # print(nan)

