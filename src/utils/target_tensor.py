import torch
import sys

sys.path.insert(1, '/home/s200640/thesis/src/')
from utils import nonMaxSuppression
import config


# Each scale has one tensor, shape of tensor is:
# shape: [num_of_anchors, cells_x, cells_y, bounding_box/anchor_data]
#        bounding_box/anchor_data format: [probability, x, y, w, h, classification]
# anchors in torch.tensor, shape: [9], unscaled anchors
class TargetTensor:
    def __init__(self, anchors: torch.Tensor, scales_cells: list):

        self.cells = scales_cells #number of cells in each scale
        self.num_of_anchors = anchors.shape[0]
        self.num_of_anchors_per_scale = self.num_of_anchors // len(self.cells)
        self.anchors = anchors.reshape(-1, 3, 2)

        self.tensor = [torch.zeros((self.num_of_anchors // 3, S, S, 6)) for S in self.cells]


    # ------------------------------------------------------
    # Computes loss with another list of tensors with given loss fcn
    def computeLossWith(self, preds: list, loss_fcn, debug=False):

        targets, d = self.tensor, torch.device(config.DEVICE)
        TargetTensor.passTargetsToDevice(preds, d)
        TargetTensor.passTargetsToDevice(targets, d)

        loss = 0
        for scale, (target, pred) in enumerate(zip(targets, preds)):

            # print(self.anchors)
            loss += loss_fcn(pred, target, self.anchors[scale, ...], debug)

        return loss


    # ------------------------------------------------------
    # Sets all tensors in target to specific device
    @staticmethod
    def passTargetsToDevice(targets: list, device: [str, torch.device]):

        for tensor in targets:

            tensor.to(device)


    # ------------------------------------------------------
    # Used to create instance from different data than original constructor,
    # anchors sould be config.ANCHORS
    @classmethod
    def fromDataLoader(cls, anchors: list, targets: list):

        anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2], dtype=torch.float64, device=config.DEVICE)
        scales_cells = [targets[i].shape[2] for i, _ in enumerate(targets)]
        tt = cls(anchors, scales_cells)
        tt.tensor = [target.detach().clone().requires_grad_(True).to(config.DEVICE) for target in targets]

        return tt


    # ------------------------------------------------------
    # Compute BBs from models predictions (iterating over all the scales)
    @staticmethod
    def computeBoundingBoxesFromPreds(preds, anchors, thresh, nms=True):

        batch_bboxes = [torch.tensor([], device=config.DEVICE) for _ in range(preds[0].shape[0])]
        for scale, pred_on_scale in enumerate(preds):

            boxes_on_scale = TargetTensor.convertCellsToBoundingBoxes(
                pred_on_scale, True, anchors[scale]
            )
            for batch_img_id, (box) in enumerate(boxes_on_scale):

                batch_bboxes[batch_img_id] = torch.cat((batch_bboxes[batch_img_id], box), dim=0)

        if nms:
            # Now for every image in batch we need to NMS
            for batch_img_id, boxes in enumerate(batch_bboxes):

                batch_bboxes[batch_img_id] = nonMaxSuppression(
                    boxes.unsqueeze(0), config.IOU_THRESHOLD, thresh
                )[0]

        return batch_bboxes


    # ------------------------------------------------------
    # Get BB from dataloader (no need to iterate over all scales)
    def getBoundingBoxesFromDataloader(self, scale):
        
        # when .tolist() returns len(bboxes[0]) = 8112 means:
        # [[(batch_img_1)[bb1], [bb2], [bb3], ...], [(batch_img_2)[bb1], [bb2], ..]]
        # without returns tensor of size [batch, num_bboxes, 6]
        # Iterate over all images in batch
        bboxes = TargetTensor.convertCellsToBoundingBoxes(self.tensor[scale], False)
        bboxes = nonMaxSuppression(bboxes, iou_thresh=1, prob_threshold=0.99)

        return bboxes


    # ------------------------------------------------------
    # Takes one tensor for one scale and returns list of all BB
    # tensor: [BATCH, A, S, S, 6] -> 6: [score, x, y, w, h, classification]
    # !!!anchor should be in scaled_anchors form
    # if thres RETURNS list of lists(each for one image in batch) containing bboxes
    @staticmethod
    def convertCellsToBoundingBoxes(tensor, fromPredictions, anchor=None, threshold=False):

        batch, num_anchors, cells = tensor.shape[0], tensor.shape[1], tensor.shape[2]

        if fromPredictions:
            assert anchor is not None, 'Anchors needed to convert to BBs from predictions!'
            tensor, _ = TargetTensor.convertPredsToBoundingBox(tensor, anchor)
            classes = torch.argmax(tensor[..., 5:], dim=-1).unsqueeze(-1)
        
        else:
            classes = tensor[..., 5:6]
        
        scores = tensor[..., 0:1]
        cell_indices = torch.arange(cells).repeat(batch, 3, cells, 1).unsqueeze(-1).to(tensor.device)
        x = (cell_indices + tensor[..., 1:2]) * (1 / cells)
        y = (cell_indices.permute(0, 1, 3, 2, 4) + tensor[..., 2:3]) * (1 / cells)
        wh = tensor[..., 3:5] / cells # This is for scaled anchors (no needed)
        # wh = tensor[..., 3:5] # This for using non-scaled anchors

        if threshold:
            bboxes = list()
            b = torch.cat((classes, scores, x, y, wh), dim=-1).reshape(batch, -1, 6)
            condition = (b[..., 1:2] >= threshold) #.reshape(1, batch, cells, cells, 1)
            condition = condition.repeat(1, 1, 1, 1, 6).reshape(batch, -1, 6)
            for idx, batch_tensor in enumerate(b):

                bboxes.append(batch_tensor[condition[idx]].reshape(-1, 6))

        else:
            bboxes = torch.cat((classes, scores, x, y, wh), dim=-1).reshape(
                batch, num_anchors * cells * cells, 6
            )

        return bboxes


    # ------------------------------------------------------
    # scaled_anchors.shape [3, 2] rquired, this is for only one scale
    @staticmethod
    def convertPredsToBoundingBox(tensor: torch.tensor, anchors: torch.tensor):

        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) 
        tensor[..., 0:3] = torch.sigmoid(tensor[..., 0:3])
        tensor[..., 3:5] = torch.exp(tensor[..., 3:5]) * anchors

        return tensor, anchors


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

        # self.scale = iou_idx // self.num_of_anchors_per_scale 
        self.scale = torch.div(iou_idx, self.num_of_anchors_per_scale, rounding_mode='floor')
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
    def setClassToCell(self, cell_x: int, cell_y: int, classification: int):

        self.tensor[self.scale][self.anchor, cell_y, cell_x, 5] = int(classification)




if __name__ == '__main__':

    # batch = 2
    # t = torch.zeros(1, batch, 3, 3, 6)
    # t[0, 0, 1, 1, ...] = 1
    # t[0, 0, 0, 0, ...] = 2
    # t[0, 1, 1, 1, ...] = 3
    # t[0, 1, 2, 1, ...] = 4
    # t[0, 1, 1, 0, ...] = 5

    # t[0, 0, 1, 1, 1] = 1
    # t[0, 0, 0, 0, 1] = 1
    # t[0, 1, 1, 1, 1] = 1
    # t[0, 1, 2, 1, 1] = 1
    # t[0, 1, 1, 0, 1] = 1

    # condition = t[..., 1] >= 0.8
    # condition = condition.reshape(1, batch, 3, 3, 1).repeat(1, 1, 1, 1, 6).reshape(batch, -1, 6)
    # # condition = condition.reshape(batch, -1, 6)

    # t = t.reshape(batch, -1, 6)

    # bboxes, batch_bboxes = list(), [torch.tensor([]) for _ in range(batch)]
    # for idx, batch_tensor in enumerate(t):

    #     bboxes.append(batch_tensor[condition[idx]].reshape(-1, 6))
    #     # bboxes[idx] += batch_tensor[condition[idx]].reshape(-1, 6).tolist()

    # for batch_img_id, (box) in enumerate(bboxes):

    #     batch_bboxes[batch_img_id] = torch.cat((batch_bboxes[batch_img_id], box), dim=0)
        
    # print(batch_bboxes)

    # from torchvision.ops import box_convert
    # xyxy = box_convert(torch.tensor([0.5, 0.5, 0.4, 0.4]), 'cxcywh', 'xyxy')
    # print(xyxy)

    # t = torch.zeros(4, 6)
    # t[..., 0] = torch.tensor([1, 2, 3, 4])
    # indices = torch.tensor([0, 2])

    # s = torch.index_select(t, dim=0, index=indices)
    # print(s)

    pass