import sys
import torch

sys.path.insert(1, '/home/mary/thesis/project/src/')
from utils import intersectionOverUnion




'''
Sources/Inspirations:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/utils.py 
https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html
'''




# ------------------------------------------------------
# bbox format: [class, prob., x, y, w, h] or [[class, prob., x1, y1, x2, y2]]
def nonMaxSuppression(
    bboxes: list, 
    iou_thresh=0.5, 
    prob_threshold=0.5, 
    midpoint_format=True
) -> list:

    # Filter all bboxes with probability under threshold score
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    final = list()
    while bboxes:

        # sort bboxes according to probability
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

        fittest_bbox = bboxes.pop(0)
        final.append(fittest_bbox)
        for box in bboxes:

            iou = intersectionOverUnion(fittest_bbox, box).item()
            bboxes.remove(box) if iou > iou_thresh else None

    return final


# ------------------------------------------------------
# This is improved version of NMS, if iou is greater than iou_thresh,
# soft nms is not removing this bbox but reducing it's probability
# for more see Navaneeth Bodlas Soft-NMS paper (link in head of this page)
def softNonMaxSuppression(
    bboxes: list, 
    iou_thresh=0.5, 
    prob_threshold=0.5, 
    midpoint_format=True
) -> list:

    # Filter all bboxes with probability under threshold score
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    final = list()
    while bboxes:

        # sort bboxes according to probability
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

        fittest_bbox = bboxes.pop(0)
        final.append(fittest_bbox)
        for box in bboxes:

            iou = intersectionOverUnion(fittest_bbox, box).item()
            box[1] = box[1] if iou < iou_thresh else box[1] * (1 - iou)

    return final


# ------------------------------------------------------
if __name__ == '__main__':

    bboxes = [[1, 0.7, 2, 2, 5, 4], [1, 0.4, 2, 2, 4.8, 4], [1, 0.9, 1.9, 1.8, 5, 4.5]]
    nms = nonMaxSuppression(bboxes, 0.5, 0.5)
    print(f'Non-max suppression: {nms}')
    soft_nms = softNonMaxSuppression(bboxes, 0.5, 0.5)
    print(f'Soft NMS: {soft_nms}')
