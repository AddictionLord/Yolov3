import torch
from torch.optim import Adam
import warnings
from tqdm import tqdm

import config
from yolo import Yolov3
from dataset import Dataset
from loss import Loss

from utils import getLoaders, TargetTensor, getBboxesToEvaluate

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
        # self.train_loader, self.val_loader = getLoaders() 
        self.train_loader = getLoaders()
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
    def saveModel(model: Yolov3, optimizer: torch.optim, path: str="./models/test_model.pth.tar"):

        print("[YOLO TRAINER]: Model saved")
        container = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": model.config,
        }

        torch.save(container, path)


    # ------------------------------------------------------
    @staticmethod
    def loadModel(path: str="./models/test_model.pth.tar"):

        print("[YOLO TRAINER]: Loading model container..")

        return torch.load(path, map_location=config.DEVICE)


    # ------------------------------------------------------
    @staticmethod
    def uploadParamsToModel(model: Yolov3, optimizer: torch.optim, params: dict):

        print("[YOLO TRAINER]: Uploading parameter to model and optimizer")
        model.load_state_dict(params['state_dict'])
        optimizer.load_state_dict(params['optimizer'])




# ------------------------------------------------------
def overfitSingleBatch(batch_size=1):

    from torch.utils.data.dataloader import DataLoader
    torch.autograd.set_detect_anomaly(True)

    val_dataset = Dataset(
        config.val_imgs_path,
        config.val_annots_path,
        config.ANCHORS,
        config.CELLS_PER_SCALE,
        config.NUM_OF_CLASSES,
        config.train_transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    model = Yolov3(config.yolo_config)
    optimizer = Adam(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fcn = Loss()
    scaler = torch.cuda.amp.GradScaler() 

    model.train()
    img, targets = next(iter(val_loader))
    losses = list()
    for epoch in range(config.NUM_OF_EPOCHS):

        print(f'epoch {epoch}/{config.NUM_OF_EPOCHS}')
        img = img.to(config.DEVICE)
        targets = TargetTensor.fromDataLoader(config.SCALED_ANCHORS, targets)
        with torch.cuda.amp.autocast():
            output = model(img)
            loss = targets.computeLossWith(output, loss_fcn)

        losses.append(loss.item())
        optimizer.zero_grad()

        # AMP scaler, vit docs. for more
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f'actual loss: {loss}, mean loss: {torch.mean(torch.tensor(losses)).item()}')




if __name__ == '__main__':

    import config
    import torch

    overfitSingleBatch(1)

    # t = YoloTrainer()
    # container = {'architecture': config.yolo_config}
    # container = YoloTrainer.loadModel('./models/stable_test.pth.tar')

    # try:
    #     t.trainYoloNet(container, load=True)

    # except KeyboardInterrupt as e:
    #     print('[YOLO TRAINER]: KeyboardInterrupt', e)

    # except Exception as e:
    #     print(e)

    # finally:
    #     saved = t.model.parameters()
    #     YoloTrainer.saveModel(t.model, t.optimizer, "./models/stable_test2.pth.tar")


    # params = YoloTrainer.loadModel("./models/test_model.pth.tar")

    # model = Yolov3(params['architecture'])
    # optimizer = Adam(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # YoloTrainer.uploadParamsToModel(model, optimizer, params)
    # loaded = model.parameters()

    # for parama, paramb in zip(saved, loaded):

    #     a = parama
    #     b = paramb

    #     if torch.rand(1).item() > 0.7:
    #         break

    # print(f'Loaded and saved parameter are same: {torch.allclose(a, b)}')
        

    







