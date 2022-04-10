import torch


'''
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/utils.py
'''


# ------------------------------------------------------
# formats: 'midpoint': [x, y, w, h]
#          'corners':  [x1, y1, x2, y2]
#          'coco':     [top_left_x, top_left_y, w, h]
# MSCOCO - 'coco' format is allowed only for loading (from dictionary)
# 
# lengths/representations: [x, y, w, h, class], [class, prob, x, y, w, h]
class BoundingBox:
    def __init__(self, bbox: [torch.tensor, list, dict], form='midpoint'):

        self.form = form

        bbox = self._parseAnnotations(bbox) if isinstance(bbox, dict) else bbox
        bbox = torch.tensor(bbox) if isinstance(bbox, list) else bbox

        bbox = self._sliceBbox(bbox)
        self._initForm(bbox)


    # ------------------------------------------------------
    # Slice bbox according to allowed representation
    def _sliceBbox(self, bbox: torch.tensor):

        if len(bbox.shape) > 1:
            rows, cols = bbox.shape[0], bbox.shape[1] 
        else:
            rows, cols = bbox.shape[0], None 

        # this is for bbox repre: [x, y, w, h]
        if rows == 4 or cols == 4:
            bbox = bbox.view(-1, 4)

        # this is for bbox repre: [x, y, w, h, class]
        elif rows == 5 or cols == 5:
            self.classification = bbox[..., 4]
            bbox = bbox[..., 0:4].view(-1, 4)

        # this is for bbox repre: [class, prob, x, y, w, h]
        elif rows == 6 or cols == 6:
            self.classification = bbox[..., 0]
            self.probability = bbox[..., 1]
            bbox = bbox[..., 2:6].view(-1, 4)

        else:
            raise("[BoundingBox]: Unknow representation, please use allowed")

        return bbox


    # ------------------------------------------------------
    def toForm(self, form: str):

        if form == 'coco':
            raise('[BoundingBox]: Coco format is only for loading, use midpoint/corners')

        elif form == 'midpoint':
            self.form = 'midpoint'
            return self.toMidpointForm()

        elif form == 'corners':
            self.form = 'corners'
            return self.toCornersForm()

        else:
            raise('[BoundingBox]: Unknow format, use midpoint/corners')
            

    # ------------------------------------------------------
    # formats: 'midpoint': [x, y, w, h]
    #          'corners':  [x1, y1, x2, y2]
    #          'coco':     [top_left_x, top_left_y, w, h]
    def _initForm(self, bbox: torch.tensor):

        if self.form == 'midpoint':
            self.bbox = bbox
            self.x, self.y = self.bbox[..., 0:1], self.bbox[..., 1:2]
            self.w, self.h = self.bbox[..., 2:3], self.bbox[..., 3:4]

        elif self.form == 'corners':
            self.bbox = bbox
            self.x1, self.y1 = self.bbox[..., 0:1], self.bbox[..., 1:2]
            self.x2, self.y2 = self.bbox[..., 2:3], self.bbox[..., 3:4]

        elif self.form == 'coco':
            self.form = 'midpoint'
            self.w, self.h = bbox[..., 2:3], bbox[..., 3:4]  
            self.x = bbox[..., 0:1] + (self.w / 2)
            self.y = bbox[..., 1:2] + (self.h / 2)
            self.bbox = torch.cat((self.x, self.y, self.w, self.h), dim=1)


    # ------------------------------------------------------
    # Parses Coco annotations
    def _parseAnnotations(self, anns: dict):

        classification = anns['category_id']
        bbox = anns['bbox']

        return torch.tensor(bbox.append(classification))


    # ------------------------------------------------------
    # If bbox is in midpoint form, this recomputes coords to corners form,
    # [x1, y1] - left top, [x2, y2] - right bottom 
    def toCornersForm(self):

        self.x1 = (self.x - self.w / 2)
        self.y1 = (self.y - self.h / 2)
        self.x2 = (self.x + self.w / 2)
        self.y2 = (self.y + self.h / 2)
        self.bbox = torch.cat((self.x1, self.y1, self.x2, self.y2), dim=1)

        return self.bbox


    # ------------------------------------------------------
    # If bbox is in corners form, this recomputes coords to midpoint form
    def toMidpointForm(self):

        self.w = (self.x1 - self.x2).abs()
        self.h = (self.y1 - self.y2).abs()
        self.x = ((self.x2 - self.x1) / 2 + self.x1)
        self.y = ((self.y2 - self.y1) / 2 + self.y1)
        self.bbox = torch.cat((self.x, self.y, self.w, self.h), dim=1)

        return self.bbox


    # # ------------------------------------------------------
    # # celss: [BATCH, A, S, S, 6] -> 6: [score, x, y, w, h, classification]
    # def _fromCellsTensor(self, cells: torch.tensor, anchors):

    #     S = cells.shape[2]
    #     batch = cells.shape[0]
    #     score = cells[..., 0:1]

    #     if anchors is not None:
    #         num_of_anchors = len(anchors)
    #         anchors = anchors.reshape(1, num_of_anchors, 1, 1, 2)
    #         cells[..., 0:2] = torch.sigmoid(cells[..., 0:2])
    #         cells[..., 3:5] = torch.exp(cells[..., 3:5]) * anchors ##Checkpoint
    #         score = torch.sigmoid(score)
    #         best_class = torch.argmax(cells[..., 5:], dim=-1).unsqueeze(-1)

    #     # cell_indices: [1, 3, 13, 13, 1]
    #     cell_indices = torch.arange(S).repeat(batch, 3, S, 1).unsqueeze(-1).to(cells.device)
    #     x = 1 / S * (cells[..., 1:2] + cell_indices)
    #     y = 1 / S * (cells[..., 2:3] + cell_indices.permute(0, 1, 3, 2, 4))
    #     wh = 1 / S * (cells[..., 3:5])

    #     self.bboxes = torch.cat((best_class, score, x, y, wh), dim=-1).reshape(
    #         batch, num_of_anchors * S * S, 6
    #     )
        
    #     return self.bboxes.tolist()


# ------------------------------------------------------
def conversionCornersMidpointTest():

    bboxes = [
        list([2, 2, 2, 4, 17]),
        list([[2, 2, 2, 4, 17], [6, 2, 2, 4, 17]]),
        torch.tensor([[2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 4]])
    ]

    for bbox in bboxes:

        b = BoundingBox(bbox, 'midpoint')
        midpoint = b.bbox
        print(midpoint)
        corners = b.toCornersForm()
        print(corners)

        b = BoundingBox(corners, 'corners')
        midpoint2 = b.toMidpointForm()
        print(midpoint2)

        b = BoundingBox(midpoint2, 'midpoint')
        corners2 = b.toCornersForm()
        print(corners2)
        print()

        assert corners[0, 1] == corners2[0, 1]
        assert corners[0, 2] == corners2[0, 2]
        assert corners[0, 3] == corners2[0, 3]

    print('Conversion test successfully passed')


# ------------------------------------------------------
if __name__ == '__main__':

    conversionCornersMidpointTest()

    # print()
    # bbox = list([2, 2, 2, 4, 1])
    # b = BoundingBox(bbox)
    # print(b.classification.item())

    #----------------------------------------

    # t = torch.arange(10) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # print(t)

    # a = t[1:5] # [1, 2, 3, 4]
    # aa = a[2:] # [3, 4]

    # print(a)
    # print(aa)

    # b = t[3:5] # [3, 4]

    # print(b)

    #----------------------------------------

    # batch = 1
    # S = 13
    # t = torch.arange(S).repeat(batch, 3, S, 1).unsqueeze(-1) #[1, 3, 13, 13, 1]

    # print(t)
    # print(t.shape)