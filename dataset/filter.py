
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np

import os 
import json




# categories = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'dog']
cat_state = {'person': 0, 'bicycle': 0, 'car': 0, 'motorcycle': 0,  'dog': 0, 'bus': 0}
categories = list(cat_state.keys())
upper_bound = 2

# annots_path = './instances_train2017.json'
annots_path = './dataset/instances_val2017.json'
filtered_annots_path = './dataset/balanced/instances_val2017.json'
RELATIVE_SRC_PATH = './dataset/val2017'
RELATIVE_DEST_PATH = './dataset/balanced/val2017'

# annots_path = './dataset/instances_train2017.json'
# filtered_annots_path = './dataset/correct_instances_train2017.json'
# RELATIVE_SRC_PATH = './dataset/train2017'
# RELATIVE_DEST_PATH = './dataset/'

coco_annots = COCO(annots_path)
    
# ---------------------------------------------------------------
def getImgIdsFromCatIds(cat_ids: list):

    img_ids = list()
    for cat_id in cat_ids:

        img_ids += coco_annots.getImgIds(catIds=cat_id)

    return img_ids


# ---------------------------------------------------------------
def getNumOfImagesInCat(categories: list):

    cat_count = dict()
    for cat_name in categories:

        cat_id = coco_annots.getCatIds(catNms=[cat_name])[0]
        img_ids = coco_annots.getImgIds(catIds=[cat_id])
        cat_count[cat_name] = len(img_ids)

    return cat_count


# ---------------------------------------------------------------
def filterDataset(cat_state: dict, upper_bound: int):

    categories = list(cat_state.keys())
    filtered_ids, filtered_info = list(), list()
    for cat_name in categories:

        img_ids = getImgIdsFromCatIds([coco_annots.getCatIds(catNms=[cat_name])[0]])
        imgs_info = coco_annots.loadImgs(img_ids)
        for img_info in imgs_info:

            if cat_state[cat_name] >= upper_bound:
                print(f'Category {cat_name} has {cat_state[cat_name]} images. ENDING')
                break

            else:
                cat_state[cat_name] += 1
                filtered_ids.append(img_info['id'])
                filtered_info.append(img_info)
                order = f"cp {RELATIVE_SRC_PATH}/{img_info['file_name']} {RELATIVE_DEST_PATH}/{img_info['file_name']}"
                os.system(order)

    filtered_anns = list()
    for ann in coco_annots.dataset['annotations']:

        if ann['image_id'] in filtered_ids and ann['iscrowd'] == 0:
            ann.pop('segmentation')
            filtered_anns.append(ann)

    data = {
        "info": coco_annots.dataset['info'],
        "licenses": coco_annots.dataset['licenses'],
        "images": filtered_info, 
        "annotations": filtered_anns,
        "categories":  coco_annots.dataset['categories']
    }

    with open(filtered_annots_path, 'w') as f:
        
        json.dump(data, f)


# ---------------------------------------------------------------
def removeFromDataset(filename: str):

    filtered_ids, filtered_info = list(), list()
    for cat_name in categories:

        img_ids = getImgIdsFromCatIds([coco_annots.getCatIds(catNms=[cat_name])[0]])
        imgs_info = coco_annots.loadImgs(img_ids)
        for img_info in imgs_info:

            if filename == img_info['file_name']:
                order = f"mv {RELATIVE_SRC_PATH}/{img_info['file_name']} {RELATIVE_DEST_PATH}"
                print(order)
                os.system(order)

            else:
                filtered_ids.append(img_info['id'])
                filtered_info.append(img_info)

    filtered_anns = list()
    for ann in coco_annots.dataset['annotations']:

        if ann['image_id'] in filtered_ids and ann['iscrowd'] == 0:
            ann.pop('segmentation')
            filtered_anns.append(ann)

    data = {
        "info": coco_annots.dataset['info'],
        "licenses": coco_annots.dataset['licenses'],
        "images": filtered_info, 
        "annotations": filtered_anns,
        "categories":  coco_annots.dataset['categories']
    }

    with open(filtered_annots_path, 'w') as f:
        
        json.dump(data, f)


# ---------------------------------------------------------------
def plotCatBalance(cat_count):

    cmap = plt.get_cmap("tab20b")

    fig = plt.figure()
    classes = cat_count.keys()
    count = cat_count.values()
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]

    # function to add value labels
    # def addlabels(x,y):
    #     for i in range(len(x)):
    #         # if i == 0:
    #         #     plt.text(i, 10000 ,y[i], ha = 'center',
    #         #             bbox = dict(facecolor = 'white', alpha = .5))

    #         # else:
    #         plt.text(i, y[i] // 2 ,y[i], ha = 'center',
    #             bbox = dict(facecolor = 'white', alpha = .5))

    # plt.bar(classes, count, color=colors)
    # plt.grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.8)
    # plt.xlabel('Classes')
    # plt.ylabel('Number of instances')
    # # plt.ylim(top=20000)

    # addlabels(list(classes), list(count))

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

        return my_autopct
    
    plt.pie(
        count, 
        labels=classes, 
        colors=colors, 
        autopct=make_autopct(count), 
        explode=[0, 0.1, 0.1, 0.1, 0.1, 0.1]
    )

    plt.tight_layout()
    plt.show()



# ---------------------------------------------------------------
if __name__ == "__main__":

    print(f'Filtering from {RELATIVE_SRC_PATH} to {RELATIVE_DEST_PATH}')
    cat_count = getNumOfImagesInCat(categories)
    print(f'cat_count before filter:\n{cat_count}')

    plotCatBalance(cat_count)



    # for k, v in cat_count.items():

    #     cat_count[k] = 1/v

    # print(f'cat_count before filter:\n{cat_count}')


    # filterDataset(cat_state, 100)
    # removeFromDataset('000000550395.jpg')

