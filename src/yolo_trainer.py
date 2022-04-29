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

Saving and loading models:
    https://www.youtube.com/watch?v=9L9jEOwRrCg

'''


class YoloTrainer:
    def __init__(self):

        self.loss = Loss()
        self.train_loader, self.val_loader = getLoaders() 
        self.scaler = torch.cuda.amp.GradScaler() 
        self.scaled_anchors = config.SCALED_ANCHORS.to(config.DEVICE)

        self.model = None
        self.optimizer = None

    
    # ------------------------------------------------------
    # Method to train specific Yolo architecture and return model
    def trainYoloNet(self, net: dict, load: bool=False):

        self.model = Yolov3(net['architecture'])
        self.optimizer = Adam(self.model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        if load:
            YoloTrainer.uploadParamsToModel(self.model, self.optimizer, net)

        for epoch in range(config.NUM_OF_EPOCHS):

            self._train(self.model, self.optimizer)
            if epoch != 0 and epoch % 4 == 0:
                self.model.eval()
                # TODO: Implement evaluating fcns
                # checkClassAccuracy()
                # preds_bboxes, target_bboxes = getBboxesToEvaluate()
                # mAP = meanAveragePrecision(preds_bboxes, target_bboxes)
                self.model.train()

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

            loader.set_postfix(loss=loss.item(), mean_loss=torch.mean(torch.tensor(losses)).item())


    # ------------------------------------------------------
    @staticmethod
    def saveModel(model: Yolov3, optimizer: torch.optim, path: str="./models/test_model.pth.tar"):

        print(f"[YOLO TRAINER]: Saving model to {path}")
        container = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": model.config,
        }

        torch.save(container, path)


    # ------------------------------------------------------
    @staticmethod
    def loadModel(path: str="./models/test_model.pth.tar"):

        print(f"[YOLO TRAINER]: Loading model container: {path}")

        return torch.load(path, map_location=config.DEVICE)


    # ------------------------------------------------------
    @staticmethod
    def uploadParamsToModel(model: Yolov3, optimizer: torch.optim, params: dict):

        print("[YOLO TRAINER]: Uploading parameters to model and optimizer")
        model.load_state_dict(params['state_dict'])
        optimizer.load_state_dict(params['optimizer'])




# ------------------------------------------------------
if __name__ == '__main__':

    import config
    import torch

    t = YoloTrainer()
    container = {'architecture': config.yolo_config}
    container = YoloTrainer.loadModel("./models/gpu_mse_loss_darknet.pth.tar")

    try:
        t.trainYoloNet(container, load=True)
        # t.trainYoloNet(container)

    except KeyboardInterrupt as e:
        print('[YOLO TRAINER]: KeyboardInterrupt', e)

    except Exception as e:
        print(e)

    finally:
        saved = t.model.parameters()
        YoloTrainer.saveModel(t.model, t.optimizer, "./models/gpu_mse_loss_darknet.pth.tar")


    # params = YoloTrainer.loadModel("./models/test_model.pth.tar")

    # model = Yolov3(params['architecture'])
    # optimizer = Adam(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # YoloTrainer.uploadParamsToModel(model, optimizer, params)
    # loaded = model.parameters()


    







