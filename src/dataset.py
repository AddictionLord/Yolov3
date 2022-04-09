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
    def _load_image(self, id: int) -> Image.Image:

        path = self.coco.loadImgs(self.ids[id])[0]["file_name"]

        return Image.open(os.path.join(self.root, path)).convert("RGB")


    # ------------------------------------------------------
    # This is pytorch fcn, see pytorch docs for more
    # Returns List where each item is an object annotation
    def _load_anns(self, id: int) -> list:

        return self.coco.loadAnns(self.coco.getAnnIds(self.ids[id]))


    # ------------------------------------------------------
    # Takes index and annotation file, calculate bbox values to
    # range (0-1), appends class id, format: [x, y, w, h, class_id]
    def _getBboxesFromAnns(self, anns: list, index: int) -> list:

        img_info = self.coco.loadImgs(self.ids[index])[0]
        width, height = img_info['width'], img_info['height']

        bboxes = list()
        for Object in anns:

            bbox = Object['bbox']
            normalized_bbox = [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height]
            normalized_bbox.append(Object['category_id'])
            bboxes.append(normalized_bbox)

        return bboxes


    # ------------------------------------------------------
    # TODO: Clean up this function, make smaller function and call them
    def __getitem__(self, index: int) -> tuple[Image.Image, tuple]:

        image = np.array(self._load_image(index))
        anns = self._load_anns(index)
        bboxes = self._getBboxesFromAnns(anns, index)

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
            x, y, width, height, classification = box # bbox = BoundingBox(box)

            # One object/Bbox corresponds to only one anchor per scale [0-2]
            bbox_has_anchor = [False, False, False]
            for iou_idx in ious_indices:

                # This finds out which scale and anchor are we handling
                scale = iou_idx // self.num_of_anchors_per_scale 
                anchor = iou_idx % self.num_of_anchors_per_scale
                
                # Computing the specific cell in grid contains the bbox midpoint
                cells = self.S[scale] # bbox.computeCell(self.S[scale])
                cell_x, cell_y = int(x * cells), int(y * cells)

                # One anchor can handle 1 object, if two objects have midpoint 
                # in same cell, another anchor box can detect it
                anchor_present = targets[scale][anchor, cell_y, cell_x, 0] 

                # If no anchor is assigned to this cell: [cell_x, cell_y] and 
                # this specific bbox has no anchor yet, we assign it:
                if not anchor_present and not bbox_has_anchor[scale]:
                    # Compute bbox: [x, y, w, h] relative to the cell
                    x, y = x * cells - cell_x, y * cells - cell_y 
                    width, height = width * cells, height * cells
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
    train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    # targets has shape tuple([BATCH, A, S, S, 6], [..], [..]) - 3 scales
    for image, targets in train_loader:

        boxes = list()
        num_of_anchors = targets[0].shape[1] 
        for i in range(num_of_anchors):

            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(targets[i].shape)
            boxes += cells_to_bboxes(
                targets[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]

        boxes = nonMaxSuppression(
            boxes, iou_threshold=1, threshold=0.7, box_format="midpoint"
        )
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)




# ------------------------------------------------------
if __name__ == '__main__':


    # data_path = 'dataset/train2017'
    # annots_path = 'dataset/instances_train2017.json'
    data_path = 'dataset/val2017'
    annots_path = 'dataset/instances_val2017.json'

    anchors = config.ANCHORS
    transform = config.test_transforms

    test()

    # d = Dataset(data_path, annots_path, anchors, transform=transform)
    # train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)

    # i = iter(train_loader)
    # image, targets = i.next()

    # print(targets[0][0, 0, :, :, 0])
    # print(targets[1][0, 0, :, :, 0])
    # print(targets[2][0, 0, :, :, 0])







