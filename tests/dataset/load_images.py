import sys
sys.path.insert(1, '/home/s200640/thesis/src/')

import torch
from torch.utils.data import DataLoader

import config
from dataset import Dataset
from utils import TargetTensor, getValLoader, getTrainLoader
from thirdparty import plot_image




# ------------------------------------------------------
# Loads couple of images and annotations from dataset
def loadImages(data_path, annots_path, subset: list=None):

    # d = Dataset(data_path, annots_path, config.ANCHORS, transform=config.test_transforms)
    # train_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False)

    # train_loader = getValLoader(subset=subset, loadbar=False)
    train_loader = getValLoader(subset=subset, loadbar=False)

    # targets has shape tuple([BATCH, A, S, S, 6], [..], [..]) - 3 scales
    for image, targets in train_loader:

        target = TargetTensor.fromDataLoader(config.ANCHORS, targets)
        bboxes = target.getBoundingBoxesFromDataloader(2)

        batch_size = image.shape[0]
        for batch_img in range(batch_size):
            
            plot_image(image[batch_img].permute(1, 2, 0).to('cpu'), bboxes[batch_img].detach().cpu())




if __name__ == '__main__':

    imgs = config.val_imgs_path
    anns = config.val_annots_path
    # config.train_imgs_path
    # config.train_annots_path

    # loadImages(imgs, anns, [25, 30, 35, 40, 45])
    loadImages(imgs, anns, [i for i in range(300, 320)])

    # 67414 is bad

