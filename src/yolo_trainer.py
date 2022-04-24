import torch
from torch.optim import Adam
import warnings
from tqdm import tqdm

import config
from yolo import Yolov3
from dataset import Dataset
from loss import Loss

from utils import getLoaders, TargetTensor

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

'''
Automatic mixed precision: 
    https://pytorch.org/docs/stable/amp.html
    https://pytorch.org/docs/stable/notes/amp_examples.html

Adam optimizer:
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

Progress bar tqdm docs:
    https://pypi.org/project/tqdm/

'''


class YoloTrainer:
    def __init__(self):

        self.loss = Loss()
        # self.train_loader, self.val_loader = getLoaders() 
        self.train_loader = getLoaders()
        self.scaler = torch.cuda.amp.GradScaler() 
        self.scaled_anchors = config.SCALED_ANCHORS.to(config.DEVICE)

        self.model = None
        self.optimizer = None

    
    # ------------------------------------------------------
    # Method to train specific Yolo architecture and return model
    def trainYoloNet(self, yolo_config: list):

        self.model = Yolov3(yolo_config)
        self.optimizer = Adam(self.model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        for epoch in range(config.NUM_OF_EPOCHS):

            self._train(self.model, self.optimizer)
            if epoch != 0 and epoch % 4 == 0:
                model.eval()
                # TODO: Implement evaluating fcns
                # checkClassAccuracy()
                # preds_bboxes, target_bboxes = getBboxesToEvaluate()
                # mAP = meanAveragePrecision(preds_bboxes, target_bboxes)
                model.train()

        return self.model, self.optimizer


    # ------------------------------------------------------
    def _train(self, model: Yolov3, optimizer: torch.optim):

        loader = tqdm(self.train_loader)
        losses = list()
        for batch, (img, targets) in enumerate(loader):

            img = img.to(config.DEVICE)
            targets = TargetTensor.fromDataLoader(self.scaled_anchors, targets)
            with torch.cuda.amp.autocast():
                output = model(img)
                loss = targets.computeLossWith(output, self.loss)

            losses.append(loss.item())
            optimizer.zero_grad()

            # AMP scaler, vit docs. for more
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            loader.set_postfix(loss=torch.mean(torch.tensor(losses)).item())


    # ------------------------------------------------------
    @staticmethod
    def saveModel(model, optimizer, filename="./models/test_model.pth.tar"):

        print("[YOLO TRAINER]: Model saved")
        container = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(container, filename)


if __name__ == '__main__':

    import config
    import torch

    t = YoloTrainer()
    t.trainYoloNet(config.yolo_config)

    # # ------------------------------------------------------
    # # Scaling anchors
    # a = torch.tensor(config.ANCHORS) # shape: [3, 3, 2]
    # S = config.CELLS_PER_SCALE # list len = 3 - need same shape, each matrix a[i, ...] to one scale
    # S = torch.tensor(S).view(-1, 1, 1).repeat(1, 3, 2)
    # scaled_anchors = a * S

    # scaled_anchors = torch.tensor(config.ANCHORS) * torch.tensor(config.CELLS_PER_SCALE).view(-1, 1, 1).repeat(1, 3, 2)
    # print(scaled_anchors[0])




