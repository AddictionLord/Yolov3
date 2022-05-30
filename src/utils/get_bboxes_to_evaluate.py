import torch
import sys
from tqdm import tqdm
from torchvision.ops import nms, box_convert



sys.path.insert(1, '/home/s200640/thesis/src/')
import config
from utils import nonMaxSuppression, TargetTensor
from thirdparty import plot_image



# ------------------------------------------------------
# Fcn takes validation loader, model and anchors and returns list of predicted
# and true bboxes to evaluate mAP and others
def getBboxesToEvaluate(
    model,
    loader,
    anchors,
    device,
    score_threshold,
):
    model.eval()
    img_id = 0
    true_bboxes, pred_bboxes = torch.tensor([], device=device), torch.tensor([], device=device)
    for batch_id, (img, targets) in enumerate(tqdm(loader)):

        img = img.to(device)
        img = img.to(torch.float32)
        t = [target.detach().clone().requires_grad_(True).to(device) for target in targets]

        batch_size = img.shape[0]
        with torch.no_grad():
            preds = model(img)

        # Inside batch_boxes each list in for one batch image
        batch_bboxes = [torch.tensor([], device=device) for _ in range(batch_size)]
        for scale, preds_on_scale in enumerate(preds):

            cells = preds[scale].shape[2] # num of cells in spec. scale
            # unnorm anchors (now in cells form) - eq to scaled_anchors
            scaled_anchors = torch.tensor([*anchors[scale]]).to(device) * cells 
            pred_boxes_scale = TargetTensor.convertCellsToBoundingBoxes(
                preds_on_scale, True, anchor=scaled_anchors, threshold=0.57
            )
            for batch_img_id, (box) in enumerate(pred_boxes_scale):

                batch_bboxes[batch_img_id] = torch.cat((batch_bboxes[batch_img_id], box), dim=0) # for each img has BBs from all scales    

        targets = TargetTensor.fromDataLoader(scaled_anchors, t)
        target_bboxes = targets.getBoundingBoxesFromDataloader(0)
        for batch_img_id, b_bboxes in enumerate(batch_bboxes):

            xyxy = box_convert(b_bboxes[..., 2:], 'cxcywh', 'xyxy')
            nms_indices = nms(xyxy, b_bboxes[..., 2], 0.1)
            nms_bboxes = torch.index_select(b_bboxes, dim=0, index=nms_indices)
            # nms_bboxes = torch.cat()
            pred_bboxes = torch.cat((pred_bboxes, nms_bboxes), dim=0)
            # for nms_bbox in nms_bboxes:
            #     pred_bboxes.append(nms_bbox.insert(0, img_id))

            targets = torch.tensor(target_bboxes[batch_img_id], device=device)
            true_bboxes = torch.cat((true_bboxes, targets), dim=0)
            # for bbox in target_bboxes[batch_img_id]:
            #     if bbox[1] > score_threshold:
            #         true_bboxes.append(bbox.insert(0, img_id))
        
            # img_id += 1

        print(batch_id)
        break
    
    model.train()

    return pred_bboxes, true_bboxes




if __name__ == '__main__':

    from yolo import Yolov3
    from dataset import Dataset
    from config import DEVICE, PROBABILITY_THRESHOLD as threshold, ANCHORS as anchors
    from yolo_trainer import YoloTrainer

    device = torch.device(DEVICE)
    transform = config.test_transforms
    val_img = config.val_imgs_path
    val_annots = config.val_annots_path

    model = Yolov3(config.yolo_config)
    model = model.to(device)
    container = YoloTrainer.loadModel('models/gpu_mse_loss.pth.tar')
    model.load_state_dict(container['state_dict'])

    d = Dataset(val_img, val_annots, anchors, transform=transform)
    # val_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(d, batch_size=config.BATCH_SIZE, shuffle=False)

    d2 = Dataset(val_img, val_annots, anchors, transform=transform)
    val_loader2 = torch.utils.data.DataLoader(d2, batch_size=config.BATCH_SIZE, shuffle=False)


    preds, true = getBboxesToEvaluate(model, val_loader, anchors.copy(), device, threshold)
    

    print(len(preds))
    print(len(true))


    