import sys
sys.path.insert(1, '/home/s200640/thesis/src/')
from torch.utils.data import DataLoader, Subset

import config
import dataset 
from tqdm import tqdm



# ------------------------------------------------------
# Return dataloader of datasets set in config file
def getLoaders(loadbar=True):

    train_dataset = dataset.Dataset(
        config.full_train_imgs_path,
        config.full_train_annots_path,
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

    val_dataset = dataset.Dataset(
        config.full_val_imgs_path,
        config.full_val_annots_path,
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

    loaders = (tqdm(train_loader), tqdm(val_loader)) if loadbar else (train_loader, val_loader)

    return loaders


# ------------------------------------------------------
def getValLoader(subset: list=None, loadbar=True):
    
    val_dataset = dataset.Dataset(
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

    return tqdm(val_loader, leave=False) if loadbar else val_loader


# ------------------------------------------------------
def getTrainLoader(subset: list=None, loadbar=True):
    
    train_dataset = dataset.Dataset(
        r'dataset/train2017',
        r'dataset/instances_train2017.json',
        config.ANCHORS,
        config.CELLS_PER_SCALE,
        config.NUM_OF_CLASSES,
        config.train_transforms,
    )

    if subset:
        train_dataset = Subset(train_dataset, subset)

    val_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return tqdm(val_loader) if loadbar else val_loader



