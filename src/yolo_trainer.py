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


    # ------------------------------------------------------
    # Method to train specific Yolo architecture and return model
    def trainYoloNet(self, yolo_config: list):

        model = Yolov3(yolo_config)
        optimizer = Adam(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        for epoch in range(config.NUM_OF_EPOCHS):

            self._train(model, optimizer)
            if epoch != 0 and epoch % 4 == 0:
                model.eval()
                # TODO: Implement evaluating fcns
                # checkClassAccuracy()
                # preds_bboxes, target_bboxes = getBboxesToEvaluate()
                # mAP = meanAveragePrecision(preds_bboxes, target_bboxes)
                model.train()

        return model

    
    # ------------------------------------------------------
    def _train(model, optimizer):

        pass





if __name__ == '__main__':

    import config
    import torch

    t = YoloTrainer()
    t.trainYoloNet(config.yolo_config)

    # ------------------------------------------------------
    # Scaling anchors
    a = torch.tensor(config.ANCHORS) # shape: [3, 3, 2]
    S = config.CELLS_PER_SCALE # list len = 3 - need same shape, each matrix a[i, ...] to one scale
    S = torch.tensor(S).view(-1, 1, 1).repeat(1, 3, 2)
    scaled_anchors = a * S

    scaled_anchors = torch.tensor(config.ANCHORS) * torch.tensor(config.CELLS_PER_SCALE).view(-1, 1, 1).repeat(1, 3, 2)



