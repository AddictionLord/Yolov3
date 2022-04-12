import torch



# Each scale has one tensor, shape of tensor is:
# shape: [num_of_anchors, cells_x, cells_y, bounding_box/anchor_data]
# bounding_box/anchor_data format: [probability, x, y, w, h, classification]
class TargetTensor:
    def __init__(self, anchors, scales_cells):

        self.anchors = anchors
        self.cells = scales_cells #number of cells in each scale

        self.num_of_anchors = anchors.shape[0]
        self.num_of_anchors_per_scale = self.num_of_anchors // len(self.cells)

        self.tensor = [torch.zeros((self.num_of_anchors // 3, S, S, 6)) for S in self.cells]


    # ------------------------------------------------------
    def __getitem__(self, index: int):

        return self.tensor[index]


    # ------------------------------------------------------
    # return number of scales
    def __len__(self):

        return len(self.cells)


    # ------------------------------------------------------
    def determineAnchorAndScale(self, iou_idx: int):

        self.scale = iou_idx // self.num_of_anchors_per_scale 
        self.anchor = iou_idx % self.num_of_anchors_per_scale

        return self.scale, self.anchor


    # ------------------------------------------------------
    def anchorIsPresent(self, cell_x: int, cell_y: int):

        return self.tensor[self.scale][self.anchor, cell_y, cell_x, 0]

    
    # ------------------------------------------------------
    def setProbabilityToCell(self, cell_x: int, cell_y: int, probability: float):

        self.tensor[self.scale][self.anchor, cell_y, cell_x, 0] = probability


    # ------------------------------------------------------
    def setBboxToCell(self, cell_x: int, cell_y: int, bbox: float):

        self.tensor[self.scale][self.anchor, cell_y, cell_x, 1:5] = bbox

    
    # ------------------------------------------------------
    def setClassToCell(self, cell_x: int, cell_y: int, classification: float):

        self.tensor[self.scale][self.anchor, cell_y, cell_x, 5] = classification

        
    




    








if __name__ == '_main_':

    pass

