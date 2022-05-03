import sys
import torch

sys.path.insert(1, '/home/mary/thesis/project/src/')
from utils import intersectionOverUnion
from torchvision.ops import nms




'''
Sources/Inspirations:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/utils.py 
https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html
https://www.youtube.com/watch?v=VAo84c1hQX8
'''




# ------------------------------------------------------
# bbox format: [class, prob., x, y, w, h] or [[class, prob., x1, y1, x2, y2]]
def nonMaxSuppression(
    bboxes: list, 
    iou_thresh=0.5, 
    prob_threshold=0.5, 
    form='midpoint'
    ) -> list:

    # Filter all bboxes with probability under threshold score
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    # sort bboxes according to probability
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    final = list()
    while bboxes:

        # sort bboxes according to probability
        # bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
        fittest_bbox = bboxes.pop(0)
        final.append(fittest_bbox)
        for box in bboxes:

            iou = intersectionOverUnion(fittest_bbox, box, form).item()
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
    form='midpoint'
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

            iou = intersectionOverUnion(fittest_bbox, box, form).item()
            box[1] = box[1] if iou < iou_thresh else box[1] * (1 - iou)

    return final




# ------------------------------------------------------
if __name__ == '__main__':

    bboxes = [
        [1, 0.7, 2, 2, 5, 4], 
        [1, 0.4, 2, 2, 4.8, 4], 
        [1, 0.9, 1.9, 1.8, 5, 4.5]
    ]

    bboxes = torch.tensor(bboxes)
    scores = bboxes[..., 1]
    bboxes = bboxes[..., 2:] / 13

    nms_boxes = nms(bboxes, scores, 0.8)
    print(nms_boxes)

    # print(scores)
    # print(bboxes)

    


    # ------------------------------------------------
    # t = torch.rand((100, 6))
    # bboxes = t.tolist()

    # nms1 = nonMaxSuppression(bboxes, 0.5, 0.5, False)
    # soft_nms = softNonMaxSuppression(bboxes, 0.5, 0.5)

    # print(f'Non-max suppression:\n{nms}\n')
    # print(f'Soft NMS:\n{soft_nms}')

    # print(nms1)

    # difference = list()
    # for _ in range(100):
    
    #     t = torch.rand((100, 6))
    #     bboxes = t.tolist()

    #     nms1 = nonMaxSuppression(bboxes, 0.5, 0.5, False)
    #     nms = non_max_suppression(bboxes, 0.5, 0.5)

    #     # print(len(nms1))
    #     # print(len(nms))

    #     difference.append(len(nms) - len(nms1))

    # difference = torch.tensor(difference, dtype=torch.float64)

    # minimum = torch.min(difference)
    # maximum = torch.max(difference)
    # mean = torch.mean(difference)
    # print(f'minimum: {minimum}\nmaximum: {maximum}\nmean: {mean}')