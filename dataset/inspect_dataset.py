#%%

#%%
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import numpy as np


# ------------------------------------------------------------------------
# Accepts tensor bbox coordinates (x, y, width, height) and
# return tuple with format (xmin, ymin, xmax, ymax)
def transformBboxCoords2xywh_topLeft(coords: torch.tensor):

    xmin, ymin, width, height = coords[0], coords[1], coords[2], coords[3]

    xmax = xmin + width
    ymax = ymin + height

    return (int(xmin), int(ymin)), int(width), int(height)


# def imshow(img, center, w, h):

#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     rect = patches.Rectangle(
#         center, w, h, linewidth=2, edgecolor='g', facecolor='none'
#     )

#     fig, ax = plt.subplots(1)

#     ax.imshow(np.transpose(npimg, (1, 2, 0)))
#     ax.add_patch(rect)
#     # plt.show()

#     return ax


def load_image(img):

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(np.transpose(npimg, (1, 2, 0)))

    return ax


def drawRectangle(ax, center, w, h):

    rect = patches.Rectangle(
        center, w, h, linewidth=2, edgecolor='g', facecolor='none'
    )
    ax.add_patch(rect)


if __name__ == '__main__':

    #data_path = 'train2017'
    #annots_path = 'instances_train2017.json'
    data_path = 'val2017'
    annots_path = 'instances_val2017.json'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    coco_train = torchvision.datasets.CocoDetection(
        root=data_path, annFile=annots_path, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        coco_train, batch_size=1, shuffle=True
    )

    while True:

        dataiter = iter(train_loader)
        image, labels = dataiter.next()

        # This remover one dimension from tensor
        # from (1, 3, 480, 640) to (3, 480, 640)
        image = torchvision.utils.make_grid(image)
        ax = load_image(image)

        for annot in labels:
            bb = torch.tensor(annot['bbox'], dtype=torch.float64)
            bot_left, w, h = transformBboxCoords2xywh_topLeft(bb)
            drawRectangle(ax, bot_left, w, h)

        plt.show()


# %%
