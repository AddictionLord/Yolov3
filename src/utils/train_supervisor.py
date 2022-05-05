import torch
import pandas

from torchmetrics.detection.mean_ap import MeanAveragePrecision

import config






# ------------------------------------------------------
class TrainSupervisor:
    def __init__(self, device):

        self.mAP = MeanAveragePrecision(box_format='cxcywh').to(device)



    # ------------------------------------------------------
    def update():

        pass