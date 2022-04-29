import torch
from torch.optim import Adam
import warnings
from tqdm import tqdm

import config
from yolo import Yolov3
from dataset import Dataset
from loss import Loss

from utils import getLoaders, TargetTensor, getBboxesToEvaluate
from thirdparty import plot_image

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
def overfitSingleBatch(batch_size: int=1, epochs: int=50, path: str='./models/batch_overfit50.pth.tar'):

    from torch.utils.data.dataloader import DataLoader
    torch.autograd.set_detect_anomaly(True)

    val_dataset = Dataset(
        config.val_imgs_path,
        config.val_annots_path,
        config.ANCHORS,
        config.CELLS_PER_SCALE,
        config.NUM_OF_CLASSES,
        config.test_transforms,
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
    for epoch in range(epochs):

        print(f'epoch {epoch}/{epochs}')
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

    print(f'Do you want to save the model to: {path} [y/n]?')
    if input() == 'y':
        YoloTrainer.saveModel(model, optimizer, path)

    return model, img




# ------------------------------------------------------
def plotDetections(model, loader, thresh, iou_thresh, anchors, preds=None):

    from torchvision.ops import nms, box_convert

    if isinstance(loader, torch.utils.data.DataLoader):
        img, targets = next(iter(loader))

    else:
        img = loader

    model.eval()
    img = img.to(config.DEVICE)
    with torch.no_grad():

        if preds is None:
            preds = model(img)

        batch_bboxes = [torch.tensor([]) for _ in range(img.shape[0])]
        for scale, pred_on_scale in enumerate(preds):

            boxes_on_scale = TargetTensor.convertCellsToBoundingBoxes(
                pred_on_scale, True, anchors[scale], thresh
            )
            for batch_img_id, (box) in enumerate(boxes_on_scale):

                batch_bboxes[batch_img_id] = torch.cat((batch_bboxes[batch_img_id], box), dim=0)

        model.train()

    for batch_img_id, b_bboxes in enumerate(batch_bboxes):

        xyxy = box_convert(b_bboxes[..., 2:6], 'cxcywh', 'xyxy')
        nms_indices = nms(xyxy, b_bboxes[..., 2], iou_thresh)
        nms_bboxes = torch.index_select(b_bboxes, dim=0, index=nms_indices)

        plot_image(img[batch_img_id].permute(1,2,0).detach().cpu(), nms_bboxes)


# ------------------------------------------------------------
def createPerfectPredictionTensor(loader):

    image, target = next(iter(loader))
    # plot_image(image[0].permute(1,2,0).detach().cpu())

    condition = (target[0][..., 0:1] == 1)
    condition = condition.repeat(1, 1, 1, 1, 6)#.reshape(batch_size, -1, 6)
    target[0][condition].reshape(-1, 6)
    # print(target[0][condition].reshape(-1, 6))

    values_idx = (target[0][..., 0:1] == 1).nonzero()
    values_idx = values_idx[..., 2:4].tolist()

    preds = target.copy()
    preds[0] = torch.zeros(1, 3, 13, 13, 11)
    preds[1] = torch.zeros(1, 3, 13, 13, 11)
    preds[2] = torch.zeros(1, 3, 13, 13, 11)
    preds[0][..., 0:3] = inv_sig(target[0][..., 0:3])
    # preds[0][..., 3:5] = torch.log(1e-16 + target[0][..., 3:5] / torch.tensor(anchors[0]).reshape(1, 3, 1, 1, 2))
    preds[0][..., 3:5] = torch.log(1e-16 + target[0][..., 3:5] / scaled_anchors[0, ...].reshape(1, 3, 1, 1, 2))
    # torch.log(1e-16 + target[..., 3:5] / anchors)
    preds[0][0, 0, 6, 7, 5] = 6
    preds[0][0, 0, 6, 8, 5] = 6


    back_target = target.copy()
    back_target[0][..., 0:3] = torch.sigmoid(preds[0][..., 0:3])
    # back_target[0][..., 3:5] = torch.exp(preds[0][..., 3:5]) * torch.tensor(anchors[0]).reshape(1, 3, 1, 1, 2)
    back_target[0][..., 3:5] = torch.exp(preds[0][..., 3:5]) * scaled_anchors[0, ...].reshape(1, 3, 1, 1, 2)
    back_target[0][..., 5] = torch.argmax(preds[0][..., 5:], dim=-1)

    # print(torch.argmax(preds[0][..., 5:], dim=-1))
    # print(target[0][0, 0, 6, 7, ...])
    # print(back_target[0][0, 0, 6, 7, ...])

    return preds




if __name__ == '__main__':

    import config
    import torch
    from yolo import Yolov3
    from dataset import Dataset
    from config import DEVICE, PROBABILITY_THRESHOLD as threshold, ANCHORS as anchors
    from yolo_trainer import YoloTrainer

    device = torch.device(DEVICE)
    transform = config.test_transforms
    val_img = config.val_imgs_path
    val_annots = config.val_annots_path    
    train_img = config.train_imgs_path
    train_annots = config.train_annots_path
    scaled_anchors = config.SCALED_ANCHORS
    batch_size = 1

    def inv_sig(x):
        return -torch.log((1 / x) - 1)

    val_dataset = Dataset(
        config.val_imgs_path,
        config.val_annots_path,
        config.ANCHORS,
        config.CELLS_PER_SCALE,
        config.NUM_OF_CLASSES,
        transform=transform
        # config.test_transforms,
    )
    img = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    val_dataset2 = Dataset(
        config.val_imgs_path,
        config.val_annots_path,
        config.ANCHORS,
        config.CELLS_PER_SCALE,
        config.NUM_OF_CLASSES,
        transform=transform
        # config.test_transforms,
    )
    img2 = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    container = YoloTrainer.loadModel('./models/batch_overfit.pth.tar')
    model = Yolov3(config.yolo_config)
    model.load_state_dict(container['state_dict'])

    model, img = overfitSingleBatch(batch_size, 50)

    plotDetections(model, img, config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, config.SCALED_ANCHORS)


    # ------------------------------------------------------------
    # Test of plotting fcns and bbox calculations 

    # preds = createPerfectPredictionTensor(loader)
    # plotDetections(model, image, config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, config.SCALED_ANCHORS, preds)



    # ------------------------------------------------------------
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
        
    




    # model = Yolov3(config.yolo_config)
    # container = YoloTrainer.loadModel('models/gpu_mse_loss.pth.tar')
    # model.load_state_dict(container['state_dict'])

    # d = Dataset(val_img, val_annots, anchors, transform=transform)
    # # val_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(d, batch_size=config.BATCH_SIZE, shuffle=False)

    # # plotDetections(model, val_loader, config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, config.SCALED_ANCHORS)

    # # img, targets = next(iter(val_loader))
    # # plotDetections(model, img, config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, config.SCALED_ANCHORS)



    







