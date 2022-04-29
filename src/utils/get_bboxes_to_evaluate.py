import torch
import sys
from tqdm import tqdm
from torchvision.ops import nms, box_convert



sys.path.insert(1, '/home/mary/thesis/project/src/')
import config
from utils import nonMaxSuppression, TargetTensor
from thirdparty import plot_image


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()
def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cpu",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(torch.device('cpu'))

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            print(len(bboxes[idx]))
            nms_boxes = nonMaxSuppression(
                bboxes[idx]
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

        break

    model.train()
    return all_pred_boxes, all_true_boxes


# # ------------------------------------------------------
# # Fcn takes validation loader, model and anchors and returns list of predicted
# # and true bboxes to evaluate mAP and others
# def getBboxesToEvaluate(
#     model,
#     loader,
#     anchors,
#     device,
#     score_threshold,
# ):
#     model.eval()
#     train_idx = 0
#     true_bboxes, pred_bboxes = list(), list()
#     for batch_id, (img, targets) in enumerate(tqdm(loader)):

#         img = img.to(device)
#         batch_size = img.shape[0]
#         with torch.no_grad():
#             preds = model(img)

#         bboxes = [[] for _ in range(batch_size)]
#         for scale, _ in enumerate(preds):
            
#             cells = preds[scale].shape[2]
#             anchor = torch.tensor([*anchors[scale]]).to(device) * cells
#             pred_boxes_scale = TargetTensor.convertCellsToBoundingBoxes(
#                 preds[scale], True, anchor
#             )
#             for scale_id, (box) in enumerate(pred_boxes_scale):

#                 bboxes[scale_id] += box            

#         target_bboxes = TargetTensor.convertCellsToBoundingBoxes(targets[2], False, anchor)
#         for bid in range(batch_size):

#             nms = nonMaxSuppression(bboxes[bid])
#             for bbox in pred_bboxes:
                
#                 pred_bboxes.append(nms.insert(0, train_id))

#             for bbox in target_bboxes[bid]:

#                 if bbox[1] > threshold:
#                     true_bboxes.append(bbox.insert(0, train_id))
        
#             train_id += 1

#         break
    
#     model.train()

#     return pred_bboxes, true_bboxes 




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
    true_bboxes, pred_bboxes = torch.tensor([]), torch.tensor([])
    for batch_id, (img, targets) in enumerate(tqdm(loader)):

        img = img.to(device)
        batch_size = img.shape[0]
        with torch.no_grad():
            preds = model(img)

        # Inside batch_boxes each list in for one batch image
        batch_bboxes = [torch.tensor([]) for _ in range(batch_size)]
        for scale, preds_on_scale in enumerate(preds):

            cells = preds[scale].shape[2] # num of cells in spec. scale
            # unnorm anchors (now in cells form) - eq to scaled_anchors
            scaled_anchors = torch.tensor([*anchors[scale]]).to(device) * cells 
            pred_boxes_scale = TargetTensor.convertCellsToBoundingBoxes(
                preds_on_scale, True, anchor=scaled_anchors, threshold=0.57
            )
            for batch_img_id, (box) in enumerate(pred_boxes_scale):

                batch_bboxes[batch_img_id] = torch.cat((batch_bboxes[batch_img_id], box), dim=0) # for each img has BBs from all scales    

        targets = TargetTensor.fromDataLoader(scaled_anchors, targets)
        target_bboxes = targets.getBoundingBoxesFromDataloader(0)
        for batch_img_id, b_bboxes in enumerate(batch_bboxes):

            xyxy = box_convert(b_bboxes[..., 2:], 'cxcywh', 'xyxy')
            nms_indices = nms(xyxy, b_bboxes[..., 2], 0.1)
            nms_bboxes = torch.index_select(b_bboxes, dim=0, index=nms_indices)
            # nms_bboxes = torch.cat()
            pred_bboxes = torch.cat((pred_bboxes, nms_bboxes), dim=0)
            # for nms_bbox in nms_bboxes:
            #     pred_bboxes.append(nms_bbox.insert(0, img_id))

            targets = torch.tensor(target_bboxes[batch_img_id])
            true_bboxes = torch.cat((true_bboxes, targets), dim=0)
            # for bbox in target_bboxes[batch_img_id]:
            #     if bbox[1] > score_threshold:
            #         true_bboxes.append(bbox.insert(0, img_id))
        
            # img_id += 1

        print(batch_id)
    
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


    