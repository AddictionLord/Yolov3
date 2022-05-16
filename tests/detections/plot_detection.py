import sys
sys.path.insert(1, '/home/s200640/thesis/src/')
from tqdm import tqdm
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import config
from yolo import Yolov3
from yolo_trainer import YoloTrainer

from dataset import Dataset
from utils import TargetTensor, getValLoader, convertDataToMAP
from thirdparty import plot_image
from perfect_tensor import createPerfectPredictionTensor







# ------------------------------------------------------
# anchors should be: torch.tensor(config.ANCHORS, dtype=torch.float16, device=device)
def plotDetections(model, loader, thresh, iou_thresh, anchors, preds=None):

    if isinstance(loader, (torch.utils.data.DataLoader, tqdm)):
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
        print(f'\nTargets shape: {btargets.shape}\n{btargets}')
        # plot_image(img[batch_img_id].permute(1,2,0).detach().cpu(), btargets.detach().cpu())
        print(f'Preds shape: {bpreds.shape}\n{bpreds}')
        plot_image(img[batch_img_id].permute(1,2,0).detach().cpu(), bpreds.detach().cpu())




if __name__ == "__main__":

    loader = getValLoader([25, 30, 35, 40, 45], False)
    # loader = getValLoader(loadbar=False)
    # container = YoloTrainer.loadModel('finetune_with_darknet')
    # container = YoloTrainer.loadModel('ultralytics_focal_loss_focal_loss_box')
    container = YoloTrainer.loadModel('overfit_again')
    
    model = Yolov3(config.yolo_config)
    model.load_state_dict(container['model'])
    model = model.to(torch.float16).to(config.DEVICE)
    anchors = torch.tensor(config.ANCHORS, dtype=torch.float16, device=config.DEVICE)


    # -------------------------------
    plotDetections(model, loader, config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, anchors)


    # # -------------------------------
    # # Test of plotDetection with generated perfect tensor
    # preds, img = createPerfectPredictionTensor(loader, anchors)
    # plotDetections(model, loader, config.PROBABILITY_THRESHOLD, config.IOU_THRESHOLD, anchors, preds)





