import torch




# ------------------------------------------------------
class BoundingBox:
    def __init__(self, bbox: [torch.tensor, list], midpoint=True):

        bbox = torch.tensor(bbox) if isinstance(bbox, list) else bbox
        bbox = bbox.view(-1, 4)

        if midpoint:
            self.midpoint = bbox
            self.corners = None
            self.x, self.y = self.midpoint[..., 0:1], self.midpoint[..., 1:2]
            self.w, self.h = self.midpoint[..., 2:3], self.midpoint[..., 3:4]

        else:
            self.corners = bbox
            self.midpoint = None
            self.x1, self.y1 = self.corners[..., 0:1], self.corners[..., 1:2]
            self.x2, self.y2 = self.corners[..., 2:3], self.corners[..., 3:4]


    # ------------------------------------------------------
    # If bbox is in midpoint form, this recomputes coords to corners form
    def toCornersForm(self):

        if self.corners is None:
            self.x1 = (self.x - self.w / 2)
            self.y1 = (self.y + self.h / 2)
            self.x2 = (self.x + self.w / 2)
            self.y2 = (self.y - self.h / 2)
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




if __name__ == '__main__':

    # bbox = list([2, 2, 2, 4])
    bbox = list([[2, 2, 2, 4], [6, 2, 2, 4]])
    # bbox = torch.tensor([[2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 4], [2, 2, 2, 4]])
    b = BoundingBox(bbox, True)
    print(b.midpoint)
    bbox2 = b.toCornersForm()
    print(bbox2)

    b = BoundingBox(bbox2, False)
    bbox3 = b.toMidpointForm()
    print(bbox3)

    b = BoundingBox(bbox3, True)
    bbox4 = b.toCornersForm()
    print(bbox4)

    # assert bbox2[1, 2] == bbox4[1, 2]





    # c = torch.tensor([1, 1])
    # b = torch.tensor([2, 2])

    # print(torch.cat((b, c), dim=-1))