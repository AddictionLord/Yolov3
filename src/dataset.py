import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
import os

import config


'''
https://pytorch.org/vision/master/_modules/torchvision/datasets/coco.html#CocoDetection.__getitem__
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
    def __getitem__(self, index: int) -> tuple[any, any]:

        image = np.array(self._load_image(index))
        anns = self._load_anns(index)
        bboxes = self._getBboxesFromAnns(anns, index)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image, bboxes = augmentations["image"], augmentations["bboxes"]



        return anns




if __name__ == '__main__':


    # data_path = 'dataset/train2017'
    # annots_path = 'dataset/instances_train2017.json'
    data_path = 'dataset/val2017'
    annots_path = 'dataset/instances_val2017.json'

    anchors = config.ANCHORS
    transform = config.test_transforms

    d = Dataset(data_path, annots_path, anchors, transform=transform)
    train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)

    i = iter(train_loader)
    bboxes = i.next()

    # print(bboxes[0].keys())






