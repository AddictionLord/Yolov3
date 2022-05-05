import sys
from torch.utils.data import DataLoader, Subset

import config
from dataset import Dataset
from tqdm import tqdm



# ------------------------------------------------------
# Return dataloader of datasets set in config file
def getLoaders():

    train_dataset = Dataset(
        config.train_imgs_path,
        config.train_annots_path,
        config.ANCHORS,
        config.CELLS_PER_SCALE,
        config.NUM_OF_CLASSES,
        config.train_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    val_dataset = Dataset(
        config.val_imgs_path,
        config.val_annots_path,
        config.ANCHORS,
        config.CELLS_PER_SCALE,
        config.NUM_OF_CLASSES,
        config.test_transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return tqdm(train_loader), tqdm(val_loader)


# ------------------------------------------------------
def getValLoader(subset: list=None):
    
    val_dataset = Dataset(
        config.val_imgs_path,
        config.val_annots_path,
        config.ANCHORS,
        config.CELLS_PER_SCALE,
        config.NUM_OF_CLASSES,
        config.test_transforms,
    )

    if subset:
        val_dataset = Subset(val_dataset, subset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    # return val_loader
    return val_loader if subset else tqdm(val_loader)





