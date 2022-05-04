import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageFile
import sys

sys.path.insert(1, '/home/s200640/thesis/src/')
import config

'''
THIRD PARTY package, sources:

[Alladin Persson]: 
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/utils.py
'''



# ------------------------------------------------------
# Author: Alladin Persson
# Bboxes in midpoint format
def plot_image(image, boxes=None):

    cmap = plt.get_cmap("tab20b")
    class_labels = config.LABELS
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image, dtype=np.float32)
    # print(im.shape)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height
    if boxes is not None:
        # Create a Rectangle patch
        for box in boxes:
            assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
            # if box[0] != 0:
            #     continue
            class_pred = box[0]
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




if __name__ == '__main__':

    from dataset import Dataset

    anchors = config.ANCHORS
    transform = config.test_transforms
    val_img = config.val_imgs_path
    val_annots = config.val_annots_path

    d = Dataset(val_img, val_annots, anchors, transform=transform)
    img, targets = d[30]
    print(len(targets))
    print(targets[0].unsqueeze(0).shape)
    plot_image(img.permute(1, 2, 0).to('cpu'))


