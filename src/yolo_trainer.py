import torch
from torch.optim import Adam
from tqdm import tqdm

from yolo import Yolov3
from dataset import Dataset
from loss import Loss
import config

from utils import getLoaders




class YoloTrainer:
    def __init__(self):

        self.loss = Loss()
        # self.train_loader, self.val_loader = getLoaders() 
        self.train_loader = getLoaders()
        self.scaler = torch.cuda.amp.GradScaler()
        self.scaled_anchors = config.SCALED_ANCHORS.to(config.DEVICE)



if __name__ == '__main__':

    import config
    import torch

    # t = YoloTrainer()
