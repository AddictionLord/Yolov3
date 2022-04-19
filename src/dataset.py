import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
import os

import config
from utils import iouBetweenBboxAnchor, nonMaxSuppression, BoundingBox, TargetTensor


'''
https://pytorch.org/vision/master/_modules/torchvision/datasets/coco.html#CocoDetection.__getitem__
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/dataset.py
https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0

'''


class Dataset(CocoDetection):
    def __init__(self, 
        root: str, annFile: str, anchors: list, img_size=416, S=[13, 26, 52], C=6,transform=None
    ):
        super(Dataset, self).__init__(root, annFile)

        self.root = root
        self.annFile = annFile

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2], dtype=torch.float64)
        self.img_size = img_size
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

        return image, tuple(targets)




# ------------------------------------------------------
# TODO: replace cells_to_bboxes, plot_image (thirdparty)
def test():

    d = Dataset(data_path, annots_path, anchors, transform=transform)
    train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    # targets has shape tuple([BATCH, A, S, S, 6], [..], [..]) - 3 scales
    for image, targets in train_loader:

        target = TargetTensor.fromDataLoader(scaled_anchors, targets)
        bboxes = target.computeBoundingBoxes(fromPredictions=False)

        plot_image(image[0].permute(1, 2, 0).to('cpu'), bboxes)


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


    t1 = torch.tensor([[True, True, False]], dtype=torch.int32)
    t2 = torch.tensor([[True, True, False]], dtype=torch.int32)
    t3 = torch.tensor([[True, True, True]], dtype=torch.int32)

    print(torch.allclose(t1, t2))
    print(torch.allclose(t1, t3))







# %%
