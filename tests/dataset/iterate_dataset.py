import sys
sys.path.insert(1, '/home/s200640/thesis/src/')

import torch
from torch.utils.data import DataLoader, Subset

import config
from dataset import Dataset
from utils import TargetTensor
from thirdparty import plot_image




# ------------------------------------------------------
# Iterate over whole dataset, test if Dataset.__get__ is ok,
# mainly ude to albumentation transforms errs
def iterateDataset(data_path, annots_path, transform, subset=None):

    dataset = Dataset(
        data_path, 
        annots_path, 
        config.ANCHORS, 
        transform=transform
    )

    if subset:
        dataset = Subset(dataset, subset)

    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False
    )

    # targets has shape tuple([BATCH, A, S, S, 6], [..], [..]) - 3 scales
    idx = start
    for image, targets in train_loader:

        # print(idx)
        idx += 1

    print(idx)




# ------------------------------------------------------
if __name__ == '__main__':

    balanced_val_imgs_path = r'dataset/balanced/val2017'
    balanced_val_imgs_path = r'dataset/balanced/instances_val2017.json'
    balanced_train_imgs_path = r'dataset/balanced/train2017'
    balanced_train_annots_path = r'dataset/balanced/instances_train2017.json'

    full_val_imgs_path = r'dataset/val2017'
    full_val_annots_path = r'dataset/instances_val2017.json'
    full_train_imgs_path = r'dataset/train2017'
    full_train_annots_path = r'dataset/instances_train2017.json'

    train_transforms = config.train_transforms
    val_transforms = config.test_transforms

    start = 0

    # iterateDataset(full_val_imgs_path, full_val_annots_path, val_transforms)
    iterateDataset(full_train_imgs_path, full_train_annots_path, train_transforms)

