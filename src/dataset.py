import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection

import config


'''
https://pytorch.org/vision/master/_modules/torchvision/datasets/coco.html#CocoDetection.__getitem__
'''


class Dataset(CocoDetection):
    def __init__(self, root, annFile, anchors, img_size=416, S=[13, 26, 52], C=6,transform=None):
        super(Dataset, self).__init__(root, annFile)

        self.root = root
        self.annFile = annFile
        self.anchors = anchors
        self.img_size = img_size
        self.S = S
        self.C = C
        self.transform = transform


    # ------------------------------------------------------
    def __getitem__(self):

        pass




if __name__ == '__main__':


    # data_path = 'dataset/train2017'
    # annots_path = 'dataset/instances_train2017.json'
    data_path = 'dataset/val2017'
    annots_path = 'dataset/instances_val2017.json'

    anchors = config.ANCHORS

    d = Dataset(data_path, annots_path, anchors)







