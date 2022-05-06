import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import warnings
from tqdm import tqdm
from pprint import pprint
import pandas

import config
from yolo import Yolov3
from dataset import Dataset
from loss import Loss

from thirdparty import plot_image
from utils import (
    getLoaders, 
    TargetTensor, 
    getBboxesToEvaluate, 
    convertDataToMAP,
    TrainSupervisor
)

# warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()
# torch.cuda.set_device('cuda:0')

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

        print(f'[YOLO TRAINER]: Training on device: {config.DEVICE}')
        self.model = Yolov3(net['architecture'])
        self.model = self.model.to(config.DEVICE)
        self.optimizer = Adam(
            self.model.parameters(), 
            config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.supervisor = TrainSupervisor(config.DEVICE, optimizer=self.optimizer)
        
        if load:
            YoloTrainer.uploadParamsToModel(
                net, self.model, self.optimizer, self.supervisor
            )

        for epoch in range(config.NUM_OF_EPOCHS):

            loss = self._train(self.model, self.optimizer)

            if epoch % 10 == 0 and epoch != 0:
                self._validate(self.model)
                print(f'epochs: {epoch}/{config.NUM_OF_EPOCHS}, mean loss: {loss}')
                mAP = self.supervisor.update(loss)
                print(self.supervisor.data[['epoch', 'loss', 'val_loss', 'map', 'map_50', 'map_75']].tail(1))

            else:
                mAP = self.supervisor.update(loss)


            if mAP > self.supervisor.best_mAP:
                self.supervisor.best_mAP = mAP
                YoloTrainer.saveModel(
                    f'{self.supervisor.name}_checkpoint', 
                    self.model, 
                    self.optimizer, 
                    self.supervisor
                )
            
        return self.model, self.optimizer


    # ------------------------------------------------------
    def _validate(self, model: Yolov3):

        model.eval()
        for batch_id, (img, targets) in enumerate(self.val_loader):

            with torch.no_grad():
                preds = model(img.to(config.DEVICE).to(torch.float32))

            anchors = torch.tensor(config.ANCHORS, dtype=torch.float16, device=config.DEVICE)
            preds_bboxes = TargetTensor.computeBoundingBoxesFromPreds(preds, anchors, config.PROBABILITY_THRESHOLD)
            targets = TargetTensor.fromDataLoader(config.ANCHORS, targets)
            target_bboxes = targets.getBoundingBoxesFromDataloader(1)

            for preds, targets in zip(preds_bboxes, target_bboxes):

                preds, targets = convertDataToMAP(preds, targets)
                self.mAP.update(preds, targets)

        model.train()
        mAP = self.mAP.compute()
        self.mAP.reset()

        return mAP


    # ------------------------------------------------------
    def _train(self, model: Yolov3, optimizer: torch.optim):

        losses = list()
        for batch, (img, targets) in enumerate(self.train_loader):

            targets = TargetTensor.fromDataLoader(config.ANCHORS, targets)
            with torch.cuda.amp.autocast():
                output = model(img.to(config.DEVICE))
                loss = targets.computeLossWith(output, self.loss)

            losses.append(loss.item())
            optimizer.zero_grad()

            # AMP scaler, see docs. for more
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            running_mean = torch.mean(torch.tensor(losses)).item()
            self.train_loader.set_postfix(loss=loss.item(), mean_loss=running_mean)

        return running_mean

    # ------------------------------------------------------
    @staticmethod
    def saveModel(
        model: Yolov3, 
        optimizer: torch.optim, 
        path: str="./models/test_model.pth.tar",
        scheduler:torch.optim.lr_scheduler=None   
    ):

        container = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": model.config,
            "scheduler": scheduler,
        }

        torch.save(container, path)
        print(f"[YOLO TRAINER]: Model saved to {path}")


    # ------------------------------------------------------
    @staticmethod
    def loadModel(path: str="./models/test_model.pth.tar"):

        print(f"[YOLO TRAINER]: Loading model container {path}")

        return torch.load(path, map_location=config.DEVICE)


    # ------------------------------------------------------
    @staticmethod
    def uploadParamsToModel(
        model: Yolov3, 
        params: dict, 
        optimizer: torch.optim=None, 
        scheduler:torch.optim.lr_scheduler.StepLR=None
    ):

        print("[YOLO TRAINER]: Uploading parameter to model and optimizer")
        model.load_state_dict(params['model'])
        if optimizer and 'optimizer' in params.keys():
            optimizer.load_state_dict(params['optimizer'])

        if scheduler and 'scheduler' in params.keys():
            scheduler.load_state_dict(params['scheduler'])


# ------------------------------------------------------
def plotDetections(model, loader, thresh, iou_thresh, anchors, preds=None):

    if isinstance(loader, torch.utils.data.DataLoader):
        img, targets = next(iter(loader))

    else:
        img = loader.to(config.DEVICE)

    model.eval()
    with torch.no_grad():

        if preds is None:
            preds = model(img.to(config.DEVICE))

        model.train()

    mAP = MeanAveragePrecision(box_format='cxcywh').to(config.DEVICE)
    anchors = torch.tensor(config.ANCHORS, dtype=torch.float16, device=config.DEVICE)
    pred_bboxes = TargetTensor.computeBoundingBoxesFromPreds(preds, anchors, thresh)
    targets = TargetTensor.fromDataLoader(config.ANCHORS, targets)
    target_bboxes = targets.getBoundingBoxesFromDataloader(1)
    for batch_img_id, (bpreds, btargets) in enumerate(zip(pred_bboxes, target_bboxes)):

        pdict, tdict = convertDataToMAP(bpreds, btargets)
        mAP.update(pdict, tdict)
        pprint(mAP.compute())
        mAP.reset()
        print(f'Targets shape: {btargets.shape}\n{btargets}')
        plot_image(img[batch_img_id].permute(1,2,0).detach().cpu(), btargets.detach().cpu())
        print(f'Targets shape: {bpreds.shape}\n{bpreds}')
        plot_image(img[batch_img_id].permute(1,2,0).detach().cpu(), bpreds.detach().cpu())



# ------------------------------------------------------
if __name__ == '__main__':

    import config
    import torch
    from yolo import Yolov3
    from dataset import Dataset
    from config import DEVICE, PROBABILITY_THRESHOLD as threshold, ANCHORS as anchors


    # ------------------------------------------------------------
    # Test of plotting fcns and bbox calculations 

    # preds = createPerfectPredictionTensor(loader)
    # plotDetections(model, image, config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, config.SCALED_ANCHORS, preds)



    # ------------------------------------------------------------
    t = YoloTrainer()
    container = {'architecture': config.yolo_config}
    # container = YoloTrainer.loadModel('./models/gpu_training_overnight.pth.tar')

    try:
        # t.trainYoloNet(container, load=True)
        t.trainYoloNet(container)

    except KeyboardInterrupt as e:
        print('[YOLO TRAINER]: KeyboardInterrupt', e)

    except Exception as e:
        print(e)

    finally:
        YoloTrainer.saveModel(t.model, t.optimizer, "./models/gpu.pth.tar")




    # t = YoloTrainer()
    # container = {'architecture': config.yolo_config}
    # # container = YoloTrainer.loadModel('./models/gpu_training_overnight.pth.tar')

    # t.trainYoloNet(container)
