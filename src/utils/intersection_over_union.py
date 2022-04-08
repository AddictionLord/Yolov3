import torch
import sys

sys.path.insert(1, '/home/mary/thesis/project/src/')
import config




# ------------------------------------------------------
# Used to find best fit (iou) of bbox with anchor boxes, position
# of bbox/anchor doesn't matter, here it's all about size (width, height)
def iouBetweenBboxAnchor(bbox, anchor):

    intersection = torch.min(bbox[..., 0], anchor[..., 0]) * torch.min(bbox[..., 1], anchor[..., 1])
    union = bbox[..., 0] * bbox[..., 1] + anchor[..., 0] * anchor[..., 1] - intersection

    return intersection / union




if __name__ == '__main__':

    l = [1, 2, 3, 4, 5]
    print(l[2:4])

    t = torch.eye(1, 2)
    print(t)
    print(t[:, 0])

    anchors = config.ANCHORS
    anch =  torch.tensor(anchors[0] + anchors[1] + anchors[2], dtype=torch.float64)
    print(anch)

    sort = torch.argsort(anch, dim=1)
    print(sort)

    box = [1, 2, 3, 4, 5]
    iou = iouBetweenBboxAnchor(torch.tensor(box[2:4]), anch)

    