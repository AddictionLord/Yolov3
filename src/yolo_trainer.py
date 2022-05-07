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
    def __init__(self, loaders=getLoaders()):

        self.loss = Loss()

        self.train_loader, self.val_loader = loaders
        # self.train_loader, self.val_loader = getValLoader([2], False), getValLoader([2], False) 

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

        loader = tqdm(self.val_loader)
        for batch_id, (img, targets) in enumerate(self.val_loader):

            val_losses = list()
            with torch.no_grad():
                preds = model(img.to(config.DEVICE).to(torch.float32))

                anchors = torch.tensor(config.ANCHORS, dtype=torch.float16, device=config.DEVICE)
                preds_bboxes = TargetTensor.computeBoundingBoxesFromPreds(preds, anchors, config.PROBABILITY_THRESHOLD)
                
                targets = TargetTensor.fromDataLoader(config.ANCHORS, targets)
                val_loss = targets.computeLossWith(preds, self.loss)
                target_bboxes = targets.getBoundingBoxesFromDataloader(1)

                self.supervisor.updateMAP(preds_bboxes, target_bboxes)
                val_losses.append(val_loss)
                running_mean = torch.mean(torch.tensor(val_losses)).item()

                loader.set_postfix(loss=val_loss.item(), mean_loss=running_mean)

        model.train()
        self.supervisor.val_loss = running_mean


    # ------------------------------------------------------
    def _train(self, model: Yolov3, optimizer: torch.optim):

        losses = list()
        loader = tqdm(self.train_loader)
        for batch, (img, targets) in enumerate(loader):

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
            loader.set_postfix(loss=loss.item(), mean_loss=running_mean)

        return running_mean


    # ------------------------------------------------------
    @staticmethod
    def saveModel(
        filename: str,
        model: Yolov3, 
        optimizer: torch.optim, 
        supervisor: TrainSupervisor=None, 
    ):

        container = {
            "filename": f'{filename}.pth.tar',
            "architecture": model.config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": supervisor.scheduler.state_dict(),
            "train_data": supervisor.state_dict(filename)
        }

        path = f'./models/{filename}.pth.tar'
        torch.save(container, path)
        print(f"[YOLO TRAINER]: Model saved to {path}")


    # ------------------------------------------------------
    @staticmethod
    def loadModel(filename: str="test_model"):

        path = f'./models/{filename}.pth.tar'
        print(f"[YOLO TRAINER]: Loading model container {path}")

        return torch.load(path, map_location=config.DEVICE)


    # ------------------------------------------------------
    @staticmethod
    def uploadParamsToModel(
        params: dict, 
        model: Yolov3, 
        optimizer: torch.optim=None, 
        supervisor: TrainSupervisor=None
    ):

        print("[YOLO TRAINER]: Uploading parameter to model and optimizer")
        model.load_state_dict(params['model'])
        if optimizer and 'optimizer' in params.keys():
            optimizer.load_state_dict(params['optimizer'])

        if supervisor and 'scheduler' in params.keys():
            supervisor.scheduler.load_state_dict(params['scheduler'])

        if supervisor and 'train_data' in params.keys():
            supervisor.load_state_dict(params['train_data'])          


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

    from config import DEVICE, PROBABILITY_THRESHOLD as threshold, ANCHORS as anchors


    # ------------------------------------------------------------
    # Test of plotting fcns and bbox calculations 

    # preds = createPerfectPredictionTensor(loader)
    # plotDetections(model, image, config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, config.SCALED_ANCHORS, preds)



    # ------------------------------------------------------------
    t = YoloTrainer()
    container = {'architecture': config.yolo_config}
    container = YoloTrainer.loadModel('supervisor')

    try:
        t.trainYoloNet(container, load=True)
        # t.trainYoloNet(container)

    except KeyboardInterrupt as e:
        print('[YOLO TRAINER]: KeyboardInterrupt', e)

    except Exception as e:
        print(e)

    finally:
        YoloTrainer.saveModel("supervisor", t.model, t.optimizer, t.supervisor)




    # t = YoloTrainer()
    # container = {'architecture': config.yolo_config}
    # container = YoloTrainer.loadModel('supervisor')

    # t.trainYoloNet(container, load=True)
