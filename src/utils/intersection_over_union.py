import torch
import sys

sys.path.insert(1, '/home/mary/thesis/project/src/')
from utils import BoundingBox
import config

''' 
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/utils.py
'''


# ------------------------------------------------------
# Used to find best fit (iou) of bbox with anchor boxes, position
# of bbox/anchor doesn't matter, here it's all about size (width, height)
def iouBetweenBboxAnchor(bbox, anchor):

    intersection = torch.min(
        bbox[..., 0], anchor[..., 0]) * torch.min(bbox[..., 1], anchor[..., 1]
    )
    union = (bbox[..., 0] * bbox[..., 1] 
        + anchor[..., 0] * anchor[..., 1] - intersection)

    return intersection / union


# ------------------------------------------------------
# Computes intersection, union and returns intersection/union
# labels/pred in format midpoint: [x, y, w, h], corners: [x1, y1, x2, y2]
def intersectionOverUnion(
    preds: torch.tensor, labels: torch.tensor, midpoint_format=True
):

    preds = BoundingBox(preds, midpoint=midpoint_format)
    labels = BoundingBox(labels, midpoint=midpoint_format)

    if midpoint_format:
        preds.toCornersForm()
        labels.toCornersForm()

    x1 = torch.max(preds.x1, labels.x1)
    y1 = torch.max(preds.y1, labels.y1)
    x2 = torch.min(preds.x2, labels.x2)
    y2 = torch.min(preds.y2, labels.y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((preds.x2 - preds.x1) * (preds.y2 - preds.y1))
    box2_area = abs((labels.x2 - labels.x1) * (labels.y2 - labels.y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)




# ------------------------------------------------------
if __name__ == '__main__':

    # l = [1, 2, 3, 4, 5]
    # print(l[2:4])

    # t = torch.eye(1, 2)
    # print(t)
    # print(t[:, 0])

    # anchors = config.ANCHORS
    # anch =  torch.tensor(anchors[0] + anchors[1] + anchors[2], dtype=torch.float64)
    # print(anch)

    # sort = torch.argsort(anch, dim=1)
    # print(sort)

    # box = [1, 2, 3, 4, 5]
    # iou = iouBetweenBboxAnchor(torch.tensor(box[2:4]), anch)

    preds = torch.tensor([[2, 2, 2, 4], [0, 0, 1, 1]])
    labels = torch.tensor([[2, 2, 3, 4], [20, 20, 1, 1]])    
    iou1 = intersectionOverUnion(preds, labels)
    print(iou1)