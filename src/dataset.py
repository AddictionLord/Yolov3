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
        super(Dataset, self).__init__(root, annFile)

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
    def _load_ann(self, id: int) -> list:

        return self.coco.loadAnns(self.coco.getAnnIds(self.ids[id]))


    # ------------------------------------------------------
    def __getitem__(self, index: int) -> tuple[any, any]:

        image = np.array(self._load_image(index))
        ann = self._load_ann(index)

        if self.transform:
            image, ann = self.target_transform(image, ann)

        return ann




if __name__ == '__main__':


    # data_path = 'dataset/train2017'
    # annots_path = 'dataset/instances_train2017.json'
    data_path = 'dataset/val2017'
    annots_path = 'dataset/instances_val2017.json'

    anchors = config.ANCHORS

    d = Dataset(data_path, annots_path, anchors)
    train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)

    i = iter(train_loader)
    bbox = i.next()
    print(type(bbox))
    print(len(bbox))






