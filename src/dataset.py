import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
import os

import config
from utils import iouBetweenBboxAnchor, nonMaxSuppression, BoundingBox, TargetTensor
from thirdparty import plot_image


'''
https://pytorch.org/vision/master/_modules/torchvision/datasets/coco.html#CocoDetection.__getitem__
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/dataset.py
https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0

'''


class Dataset(CocoDetection):
    def __init__(self, 
        root: str, annFile: str, anchors: list, S=[13, 26, 52], C=6,transform=None
    ):
        super(Dataset, self).__init__(root, annFile)

        self.root = root
        self.annFile = annFile

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2], dtype=torch.float64)
        self.S = S
        self.C = C
        self.transform = transform

        self.iou_thresh = 0.5


    # ------------------------------------------------------
    # Returns list of BoundingBox instances
    def _parseAnnotations(self, anns: dict, index: int) -> list:

        img_info = self.coco.loadImgs(self.ids[index])[0]
        bboxes = list()
        for Object in anns:

            bbox = BoundingBox(Object, form='coco')
            bbox.normalize(img_info['width'], img_info['height'])
            bboxes.append(bbox.toTransform())

        return bboxes


    # ------------------------------------------------------
    # Used by train loader to load images and targets annots with anchors
    def __getitem__(self, index: int) -> tuple[Image.Image, tuple]:

        image, anns = super().__getitem__(index)
        bboxes = self._parseAnnotations(anns, index)
        image = np.array(image)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image, bboxes = augmentations["image"], augmentations["bboxes"]

        # 6 -> objectness score, x, y, w, h, classification
        targets = TargetTensor(self.anchors, self.S)
        for box in bboxes:

            # Finding best fit for bbox/anchor
            ious = iouBetweenBboxAnchor(torch.tensor(box[2:4]), self.anchors)
            ious_indices = torch.argsort(ious, dim=0, descending=True)
            bbox = BoundingBox(list(box))
            # One object/Bbox corresponds to only one anchor per scale [0-2]
            bbox_has_anchor = [False, False, False] # 3 bools for 3 scales
            for iou_idx in ious_indices:

                # This finds out which scale and anchor are we handling
                scale, _ = targets.determineAnchorAndScale(iou_idx)
                # Computing the specific cell in grid contains the bbox midpoint
                cx, cy = bbox.computeCells(self.S[scale])
                # One anchor can handle 1 object, if two objects have midpoint 
                # in same cell, another anchor box can detect it
                anchor_present = targets.anchorIsPresent(cx, cy)
                # If no anchor is assigned to this cell: [cx - cell_x, cy - cell_y] and 
                # this specific bbox has no anchor yet, we assign it:
                if not anchor_present and not bbox_has_anchor[scale]:
                    # Assign all the data to target tensor
                    targets.setProbabilityToCell(cx, cy, 1)
                    targets.setBboxToCell(cx, cy, bbox.bb_cell_relative)
                    targets.setClassToCell(cx, cy, bbox.classification)
                    bbox_has_anchor[scale] = True

                elif not anchor_present and ious[iou_idx] > self.iou_thresh:
                    targets.setProbabilityToCell(cx, cy, -1)

        return image.to(torch.float16), tuple(targets.tensor)




# ------------------------------------------------------
# Loads couple of images and annotations from dataset
def test(data_path, annots_path):

    d = Dataset(data_path, annots_path, anchors, transform=transform)
    train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)

    # targets has shape tuple([BATCH, A, S, S, 6], [..], [..]) - 3 scales
    for image, targets in train_loader:

        target = TargetTensor.fromDataLoader(config.ANCHORS, targets)
        bboxes = target.getBoundingBoxesFromDataloader(2)

        batch_size = image.shape[0]
        for batch_img in range(batch_size):
            
            plot_image(image[batch_img].permute(1, 2, 0).to('cpu'), bboxes[batch_img])





# ------------------------------------------------------
if __name__ == '__main__':

    anchors = config.ANCHORS
    transform = config.test_transforms

    val_img = config.val_imgs_path
    val_annots = config.val_annots_path

    test(val_img, val_annots)

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


    t1 = torch.tensor([[True, True, False]], dtype=torch.int32)
    t2 = torch.tensor([[True, True, False]], dtype=torch.int32)
    t3 = torch.tensor([[True, True, True]], dtype=torch.int32)

    print(torch.allclose(t1, t2))
    print(torch.allclose(t1, t3))


