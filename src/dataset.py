import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
import os

import config
from utils import iouBetweenBboxAnchor, nonMaxSuppression, BoundingBox


'''
https://pytorch.org/vision/master/_modules/torchvision/datasets/coco.html#CocoDetection.__getitem__
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/dataset.py
https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0

'''


class Dataset(CocoDetection):
    def __init__(self, 
        root: str, annFile: str, anchors: list, img_size=416, S=[13, 26, 52], C=6,transform=None
    ):
        super(Dataset, self).__init__(root, annFile, transform)

        self.root = root
        self.annFile = annFile

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2], dtype=torch.float64)
        self.num_of_anchors = self.anchors.shape[0]
        self.num_of_anchors_per_scale = self.num_of_anchors // len(S)

        self.img_size = img_size
        self.S = S
        self.C = C
        self.transform = transform

        self.iou_thresh = 0.5


    # ------------------------------------------------------
    # This is pytorch fcn, see pytorch docs for more
    # Returns Image.Image object
    def _load_image(self, index: int) -> Image.Image:

        self.img_info = self.coco.loadImgs(self.ids[index])[0]
        path = self.img_info["file_name"]

        return Image.open(os.path.join(self.root, path)).convert("RGB")


    # ------------------------------------------------------
    # This is pytorch fcn, see pytorch docs for more
    # Returns List where each item is an object annotation
    def _load_anns(self, index: int) -> list:

        return self.coco.loadAnns(self.coco.getAnnIds(self.ids[index]))


    # ------------------------------------------------------
    # Takes index and annotation file, normalize bbox values to
    # range (0-1), appends class id, returns format: [x, y, w, h, class_id]
    def _getBboxesFromAnns(self, anns: list, index: int) -> list:

        width, height = self.img_info['width'], self.img_info['height']
        bboxes = list()
        for Object in anns:

            bbox = transformBboxCoords(torch.tensor(Object['bbox']))
            print(Object['bbox'])
            normalized_bbox = [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height]
            normalized_bbox.append(Object['category_id'])
            bboxes.append(normalized_bbox)

        return bboxes


    # ------------------------------------------------------
    # TODO: Clean up this function, make smaller function and call them
    # use super().__getitem__(index) probably
    def __getitem__(self, index: int) -> tuple[Image.Image, tuple]:

        image = np.array(self._load_image(index))
        anns = self._load_anns(index)
        bboxes = self._getBboxesFromAnns(anns, index)

        plot_im(image, bboxes)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image, bboxes = augmentations["image"], augmentations["bboxes"]


        # 6 -> objectness score, x, y, w, h, classification
        targets = [torch.zeros((self.num_of_anchors // 3, S, S, 6)) for S in self.S]
        # Loop through all bboxes in the image and find best anchor
        for box in bboxes:

            # Finding best fit for bbox/anchor
            ious = iouBetweenBboxAnchor(torch.tensor(box[2:4]), self.anchors)
            ious_indices = torch.argsort(ious, dim=0, descending=True)
            ix, iy, iwidth, iheight, classification = box # bbox = BoundingBox(box)

            # One object/Bbox corresponds to only one anchor per scale [0-2]
            bbox_has_anchor = [False, False, False] # 3 bools for 3 scales
            for iou_idx in ious_indices:

                # This finds out which scale and anchor are we handling
                scale = iou_idx // self.num_of_anchors_per_scale 
                anchor = iou_idx % self.num_of_anchors_per_scale
                
                # Computing the specific cell in grid contains the bbox midpoint
                cells = self.S[scale] # bbox.computeCell(self.S[scale])
                cell_x, cell_y = int(ix * cells), int(iy * cells)

                # One anchor can handle 1 object, if two objects have midpoint 
                # in same cell, another anchor box can detect it
                anchor_present = targets[scale][anchor, cell_y, cell_x, 0] 

                # If no anchor is assigned to this cell: [cell_x, cell_y] and 
                # this specific bbox has no anchor yet, we assign it:
                if not anchor_present and not bbox_has_anchor[scale]:
                    # Compute bbox: [x, y, w, h] relative to the cell
                    x, y = ix * cells - cell_x, iy * cells - cell_y 
                    width, height = iwidth * cells, iheight * cells
                    bbox = torch.tensor([x, y, width, height]) 

                    # Assign all the data to target tensor
                    targets[scale][anchor, cell_y, cell_x, 0] = 1
                    targets[scale][anchor, cell_y, cell_x, 1:5] = bbox #bbox.relativeToCell()
                    targets[scale][anchor, cell_y, cell_x, 5] = classification
                    bbox_has_anchor[scale] = True

                elif not anchor_present and ious[iou_idx] > self.iou_thresh:
                    targets[scale][anchor, cell_y, cell_x, 0] = -1

        return image, tuple(targets)




# ------------------------------------------------------
def test():

    d = Dataset(data_path, annots_path, anchors, transform=transform)
    train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    # targets has shape tuple([BATCH, A, S, S, 6], [..], [..]) - 3 scales
    for image, targets in train_loader:

        num_of_anchors = targets[0].shape[1] 
        boxes = list()
        for i in range(num_of_anchors):

            anchor = scaled_anchors[i]

            # boxes += BoundingBox(targets[i], True, anchor).bboxes.tolist()[0]
            boxes += cells_to_bboxes(
                targets[i], is_preds=False, S=targets[i].shape[2], anchors=anchor
            )[0]
            
            # boxes = nonMaxSuppression(boxes, 0.6, 0.65, True)
            boxes = nonMaxSuppression(boxes, 1, 0.7, True)
        
        print(torch.tensor(boxes))            
        print(len(boxes))            
        plot_image(image[0].permute(1, 2, 0).to('cpu'), boxes)


def atest():
    anchors = config.ANCHORS

    transform = config.test_transforms

    d = Dataset(data_path, annots_path, anchors, transform=transform)
    loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = non_max_suppression(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(len(boxes))
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# ------------------------------------------------------
# Bboxes in midpoint format
def plot_image(image, boxes):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    cmap = plt.get_cmap("tab20b")
    class_labels = config.LABELS
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    # print(im.shape)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0] - 1
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


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



# ------------------------------------------------------------------------
# Accepts tensor bbox coordinates (top_left_x, top_left_y, width, height) and
# return tuple with format (x, y, w, h)
def transformBboxCoords(coords: torch.tensor):

    tlx, tly, width, height = coords[0], coords[1], coords[2], coords[3]

    x = tlx + width / 2
    y = tly + height / 2

    return [int(x), int(y), int(width), int(height)]



# bboxes in midpoint format [x, y, w, h, class]
def plot_im(image, bboxes):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    cmap = plt.get_cmap("tab20b")
    class_labels = config.LABELS
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]

    im = np.array(image)
    height, width, _ = im.shape
    ig, ax = plt.subplots(1)
    ax.imshow(im)
    for b in bboxes:
        box = b.copy()

        classification = box[-1] - 1
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(classification)],
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(classification)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(classification)], "pad": 0},
        )

    plt.show()


# ------------------------------------------------------
if __name__ == '__main__':


    # data_path = 'dataset/train2017'
    # annots_path = 'dataset/instances_train2017.json'
    data_path = 'dataset/val2017'
    annots_path = 'dataset/instances_val2017.json'

    anchors = config.ANCHORS
    transform = config.test_transforms



    # atest()
    test()

    # d = Dataset(data_path, annots_path, anchors, transform=None)
    # train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True)

    # for image, bboxes in train_loader:

    #     bboxes = torch.tensor(bboxes).tolist()
    #     plot_im(image[0].to("cpu"), bboxes)

    # i = iter(train_loader)
    # image, bboxes = i.next()

    # bboxes = torch.tensor(bboxes).tolist()
    # plot_im(image[0].to("cpu"), bboxes)



    # print(targets[0][0, 0, :, :, 0])
    # print(targets[1][0, 0, :, :, 0])
    # print(targets[2][0, 0, :, :, 0])

    # index = 8

    # image = np.array(d._load_image(index))
    # anns = d._load_anns(index)
    # bboxes = d._getBboxesFromAnns(anns, index)

    # plot_im(image, bboxes)








# %%
