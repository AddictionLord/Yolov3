import torch

from utils import nonMaxSuppression


# Each scale has one tensor, shape of tensor is:
# shape: [num_of_anchors, cells_x, cells_y, bounding_box/anchor_data]
# bounding_box/anchor_data format: [probability, x, y, w, h, classification]
class TargetTensor:

    # @dispatch(torch.Tensor, list)
    def __init__(self, anchors: torch.Tensor, scales_cells: list):

        self.anchors = anchors
        self.cells = scales_cells #number of cells in each scale

        self.num_of_anchors = anchors.shape[0]
        self.num_of_anchors_per_scale = self.num_of_anchors // len(self.cells)

        self.tensor = [torch.zeros((self.num_of_anchors // 3, S, S, 6)) for S in self.cells]


    # ------------------------------------------------------
    # Used to create instance from different data than original constructor
    @classmethod
    def fromDataLoader(cls, scaled_anchors: list, targets: list):

        scales_cells = [targets[i].shape[2] for i, _ in enumerate(targets)]
        tt = cls(scaled_anchors.reshape(-1, 2), scales_cells)
        tt.anchors, tt.tensor = scaled_anchors, targets

        return tt


    # ------------------------------------------------------
    # Difference between computing BBs from predictions and from dataloader
    def computeBoundingBoxes(self, fromPredictions=True):

        num_of_anchors = self.num_of_anchors_per_scale
        bboxes = list()
        for scale in range(len(self.cells)):

            anchor = self.anchors[scale]
            bboxes += TargetTensor.convertCellsToBoundingBoxes(
                self.tensor[scale], fromPredictions, anchor
            )
            bboxes = nonMaxSuppression(bboxes, 1, 0.7)

        return bboxes


    # ------------------------------------------------------
    # Takes one tensor on scale and returns list of all BB
    # tensor: [BATCH, A, S, S, 6] -> 6: [score, x, y, w, h, classification]
    @staticmethod
    def convertCellsToBoundingBoxes(tensor, fromPredictions, anchor):

        batch, anchors, cells = tensor.shape[0], tensor.shape[1], tensor.shape[2]
        scores, classes = tensor[..., 0:1], tensor[..., 5:6]

        # if fromPredictions:
        
        cell_indices = torch.arange(cells).repeat(batch, 3, cells, 1).unsqueeze(-1).to(tensor.device)
        x = (cell_indices + tensor[..., 1:2]) * (1 / cells)
        y = (cell_indices.permute(0, 1, 3, 2, 4) + tensor[..., 2:3]) * (1 / cells)
        wh = tensor[..., 3:5] / cells

        return torch.cat((classes, scores, x, y, wh), dim=-1).reshape(
            batch, anchors * cells * cells, 6).tolist()[0]


    # ------------------------------------------------------
    def __getitem__(self, index: int):

        return self.tensor[index]


    # ------------------------------------------------------
    # return number of scales
    def __len__(self):

        return len(self.cells)


    # ------------------------------------------------------
    # 3 anchors boxes in each of 3 scales = 9 total anchor boxes
    # iou_idx is index of bbox from argsorted iou(bbox, anchor) scores
    def determineAnchorAndScale(self, iou_idx: int):

        self.scale = iou_idx // self.num_of_anchors_per_scale 
        self.anchor = iou_idx % self.num_of_anchors_per_scale

        return self.scale, self.anchor


    # ------------------------------------------------------
    # Returns 0 or 1 according to anchors box presence (1 = present)
    def anchorIsPresent(self, cell_x: int, cell_y: int) -> int:

        return self.tensor[self.scale][self.anchor, cell_y, cell_x, 0]

    
    # ------------------------------------------------------
    # Takes cell position index (x, y) and sets probability of object presence to it
    def setProbabilityToCell(self, cell_x: int, cell_y: int, probability: float):

        self.tensor[self.scale][self.anchor, cell_y, cell_x, 0] = probability


    # ------------------------------------------------------
    # Takes cell position index (x, y) and sets bbox values to it
    def setBboxToCell(self, cell_x: int, cell_y: int, bbox: float):

        self.tensor[self.scale][self.anchor, cell_y, cell_x, 1:5] = bbox

    
    # ------------------------------------------------------
    # Takes cell position index (x, y) and sets which class does it belong to
    def setClassToCell(self, cell_x: int, cell_y: int, classification: float):

        self.tensor[self.scale][self.anchor, cell_y, cell_x, 5] = classification

        
    




    








if __name__ == '_main_':

    pass

