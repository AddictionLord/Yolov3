import torch


'''
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/utils.py
'''


# ------------------------------------------------------
# TODO: Clean the constructor, make some format handling private fcns
class BoundingBox:
    def __init__(self, 
        bbox: [torch.tensor, list], 
        midpoint=True, 
        anchors:torch.tensor = None
    ):

        bbox = torch.tensor(bbox) if isinstance(bbox, list) else bbox
        # tensor shape: [BATCH, A, S, S, 6]
        if len(bbox.shape) == 5:
            self._fromCellsTensor(bbox, anchors)

        # this is for format bbox: [x, y, w, h, class]
        elif bbox.shape[0] == 5:
            self.classification = bbox[..., 4]
            bbox = bbox[..., 0:4].view(-1, 4)

        # this is for format bbox: [class, prob, x, y, w, h]
        elif bbox.shape[0] == 6:
            self.classification = bbox[..., 0]
            self.probability = bbox[..., 1]
            bbox = bbox[..., 2:6].view(-1, 4)

        if midpoint:
            self.midpoint, self.corners = bbox, None
            self.x, self.y = self.midpoint[..., 0:1], self.midpoint[..., 1:2]
            self.w, self.h = self.midpoint[..., 2:3], self.midpoint[..., 3:4]

        else:
            self.midpoint, self.corners = None, bbox
            self.x1, self.y1 = self.corners[..., 0:1], self.corners[..., 1:2]
            self.x2, self.y2 = self.corners[..., 2:3], self.corners[..., 3:4]


    # ------------------------------------------------------
    # celss: [BATCH, A, S, S, 6] -> 6: [score, x, y, w, h, classification]
    def _fromCellsTensor(self, cells: torch.tensor, anchors):

        S = cells.shape[2]
        batch = cells.shape[0]
        score = cells[..., 0:1]

        if anchors is not None:
            num_of_anchors = len(anchors)
            anchors = anchors.reshape(1, num_of_anchors, 1, 1, 2)
            cells[..., 0:2] = torch.sigmoid(cells[..., 0:2])
            cells[..., 3:5] = torch.exp(cells[..., 3:5]) * anchors ##Checkpoint
            score = torch.sigmoid(score)
            best_class = torch.argmax(cells[..., 5:], dim=-1).unsqueeze(-1)

        # cell_indices: [1, 3, 13, 13, 1]
        cell_indices = torch.arange(S).repeat(batch, 3, S, 1).unsqueeze(-1).to(cells.device)
        x = 1 / S * (cells[..., 1:2] + cell_indices)
        y = 1 / S * (cells[..., 2:3] + cell_indices.permute(0, 1, 3, 2, 4))
        wh = 1 / S * (cells[..., 3:5])

        self.bboxes = torch.cat((best_class, score, x, y, wh), dim=-1).reshape(
            batch, num_of_anchors * S * S, 6
        )
        
        return self.bboxes.tolist()


    # ------------------------------------------------------
    # If bbox is in midpoint form, this recomputes coords to corners form,
    # [x1, y1] - left top, [x2, y2] - right bottom 
    def toCornersForm(self):

        if self.corners is None:
            self.x1 = (self.x - self.w / 2)
            self.y1 = (self.y - self.h / 2)
            self.x2 = (self.x + self.w / 2)
            self.y2 = (self.y + self.h / 2)
            self.corners = torch.cat((self.x1, self.y1, self.x2, self.y2), dim=1)

        return self.corners


    # ------------------------------------------------------
    # If bbox is in corners form, this recomputes coords to midpoint form
    def toMidpointForm(self):

        if self.midpoint is None:
            self.w = (self.x1 - self.x2).abs()
            self.h = (self.y1 - self.y2).abs()
            self.x = ((self.x2 - self.x1) / 2 + self.x1)
            self.y = ((self.y2 - self.y1) / 2 + self.y1)
            self.midpoint = torch.cat((self.x, self.y, self.w, self.h), dim=1)

        return self.midpoint




# ------------------------------------------------------
if __name__ == '__main__':

    # # bbox = list([2, 2, 2, 4])
    # bbox = list([[2, 2, 2, 4], [6, 2, 2, 4]])
    # # bbox = torch.tensor([[2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 4]])
    # b = BoundingBox(bbox, True)
    # print(b.midpoint)
    # bbox2 = b.toCornersForm()
    # print(bbox2)

    # b = BoundingBox(bbox2, False)
    # bbox3 = b.toMidpointForm()
    # print(bbox3)

    # b = BoundingBox(bbox3, True)
    # bbox4 = b.toCornersForm()
    # print(bbox4)

    # assert bbox2[1, 2] == bbox4[1, 2]

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

    batch = 1
    S = 13
    t = torch.arange(S).repeat(batch, 3, S, 1).unsqueeze(-1) #[1, 3, 13, 13, 1]

    print(t)
    print(t.shape)