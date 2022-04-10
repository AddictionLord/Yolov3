import sys
import torch

sys.path.insert(1, '/home/mary/thesis/project/src/')
from utils import intersectionOverUnion




'''
Sources/Inspirations:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/utils.py 
https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html
https://www.youtube.com/watch?v=VAo84c1hQX8
'''




# ------------------------------------------------------
# bbox format: [class, prob., x, y, w, h] or [[class, prob., x1, y1, x2, y2]]
#  TODO: Check precision NMS, some differences was seen with A. Persson nms fcn
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

# def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
#     """
#     Video explanation of this function:
#     https://youtu.be/XXYG5ZWtjj0
#     This function calculates intersection over union (iou) given pred boxes
#     and target boxes.
#     Parameters:
#         boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
#         boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
#         box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
#     Returns:
#         tensor: Intersection over union for all examples
#     """

#     if box_format == "midpoint":
#         box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
#         box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
#         box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
#         box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
#         box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
#         box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
#         box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
#         box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

#     if box_format == "corners":
#         box1_x1 = boxes_preds[..., 0:1]
#         box1_y1 = boxes_preds[..., 1:2]
#         box1_x2 = boxes_preds[..., 2:3]
#         box1_y2 = boxes_preds[..., 3:4]
#         box2_x1 = boxes_labels[..., 0:1]
#         box2_y1 = boxes_labels[..., 1:2]
#         box2_x2 = boxes_labels[..., 2:3]
#         box2_y2 = boxes_labels[..., 3:4]

#     x1 = torch.max(box1_x1, box2_x1)
#     y1 = torch.max(box1_y1, box2_y1)
#     x2 = torch.min(box1_x2, box2_x2)
#     y2 = torch.min(box1_y2, box2_y2)

#     intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
#     box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
#     box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

#     return intersection / (box1_area + box2_area - intersection + 1e-6)

# def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
#     """
#     Video explanation of this function:
#     https://youtu.be/YDkjWEN8jNA
#     Does Non Max Suppression given bboxes
#     Parameters:
#         bboxes (list): list of lists containing all bboxes with each bboxes
#         specified as [class_pred, prob_score, x1, y1, x2, y2]
#         iou_threshold (float): threshold where predicted bboxes is correct
#         threshold (float): threshold to remove predicted bboxes (independent of IoU)
#         box_format (str): "midpoint" or "corners" used to specify bboxes
#     Returns:
#         list: bboxes after performing NMS given a specific IoU threshold
#     """

#     assert type(bboxes) == list

#     bboxes = [box for box in bboxes if box[1] > threshold]
#     bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
#     bboxes_after_nms = []

#     while bboxes:
#         chosen_box = bboxes.pop(0)

#         bboxes = [
#             box
#             for box in bboxes
#             if box[0] != chosen_box[0]
#             or intersection_over_union(
#                 torch.tensor(chosen_box[2:]),
#                 torch.tensor(box[2:]),
#                 box_format=box_format,
#             )
#             < iou_threshold
#         ]

#         bboxes_after_nms.append(chosen_box)

#     return bboxes_after_nms


# ------------------------------------------------------
if __name__ == '__main__':

    bboxes = [
        [1, 0.7, 2, 2, 5, 4], 
        [1, 0.4, 2, 2, 4.8, 4], 
        [1, 0.9, 1.9, 1.8, 5, 4.5]
    ]

    t = torch.rand((100, 6))
    bboxes = t.tolist()

    nms1 = nonMaxSuppression(bboxes, 0.5, 0.5, False)
    # soft_nms = softNonMaxSuppression(bboxes, 0.5, 0.5)

    # print(f'Non-max suppression:\n{nms}\n')
    # print(f'Soft NMS:\n{soft_nms}')

    print(nms1)

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