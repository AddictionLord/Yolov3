import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

'''
https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0

'''



# ------------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 416
NUM_WORKERS = 4
BATCH_SIZE = 1
CELLS_PER_SCALE = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
NUM_OF_CLASSES = 6
PIN_MEMORY = True
NUM_OF_EPOCHS = 15000
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
LAMBDA_COORD = 10 #10
LAMBDA_NOOBJ = 0.5 #10

PROBABILITY_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4




# ------------------------------------------------------
# Paths
# val_imgs_path = r'dataset/balanced/val2017'
# val_annots_path = r'dataset/balanced/instances_val2017.json'
train_imgs_path = r'dataset/balanced/train2017'
train_annots_path = r'dataset/balanced/instances_train2017.json'

val_imgs_path = r'dataset/val2017'
val_annots_path = r'dataset/instances_val2017.json'
# train_imgs_path = r'dataset/train2017'
# train_annots_path = r'dataset/instances_train2017.json'


darknet53_path = 'models/pretrained/darknet53.conv.74'


# ------------------------------------------------------
# Anchors computed bz K-means for MSCoco dataset
# Each list inside of ANCHORS correspond to specific prediction scale (3 scales)
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

SCALED_ANCHORS =  (
    torch.tensor(ANCHORS) * 
    torch.tensor(CELLS_PER_SCALE).view(-1, 1, 1).repeat(1, 3, 2)
).to(torch.device(DEVICE))


# ------------------------------------------------------
# Transformation for train and test datasets
scale = 1.2
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)


# ------------------------------------------------------
# Original (Redmons) yolo config (without darknet53)
yolo_config = [
    # Darknet-53 before yolo
    (512, 1, 1),
    (1024, 3, 1),
    ["C", 1],
    (512, 1, 1),
    "S",
    (256, 1, 1),
    {"U": 2},
    (256, 1, 1),
    (512, 3, 1),
    ["C", 1],
    (256, 1, 1),
    "S",
    (128, 1, 1),
    {"U": 2},
    (128, 1, 1),
    (256, 3, 1),
    ["C", 1],
    (128, 1, 1),
    "S"
]

darknet_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["R", 1],
    (128, 3, 2),
    ["R", 2],
    (256, 3, 2),
    ["R", 8],
    (512, 3, 2),
    ["R", 8],
    (1024, 3, 2),
    ["R", 4]
]

# ------------------------------------------------------
# Modified MSCoco dataset labels
LABELS = [
 'person',
 'bicycle',
 'car',
 'motorcycle',
 'dog',
 'bus'
]

# MSCoco dataset labels
COCO_LABELS = [
 'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'horse',
 'dog',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]

# Connects COCO_LABLES with LABELS to train custom dataset 
LABELS_INDICES = [COCO_LABELS.index(label) for label in LABELS]




if __name__ == '__main__':

    # t = torch.tensor([0, 1, 2, 3, 5, 17, 17, 17])
    # print(t)
    # seventeen = torch.where(t == 17)
    # t[seventeen] = 4
    # print(t)


    # indices = list()
    # for label in LABELS:

    #     indices.append(COCO_LABELS.index(label))

    # indices = [COCO_LABELS.index(label) for label in LABELS] 
    # print(indices)

    # print(LABELS_INDICES.index(17))


    # t = torch.tensor(ANCHORS, device=DEVICE)

    # t = torch.tensor(ANCHORS[0] + ANCHORS[1] + ANCHORS[2], dtype=torch.float64)
    # print(t)

    print(torch.tensor(CELLS_PER_SCALE).view(-1, 1, 1).repeat(1, 3, 2))

