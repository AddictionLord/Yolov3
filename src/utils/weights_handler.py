'''
Loading weights from PJ Redmon's darknet53.conv.74 file
Inspiration/sources: 
    (darknet53.conv.74) https://pjreddie.com/media/files/darknet53.conv.74
    (Erik Lindernoren) https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/e54d52e6500b7bbd687769f7abfd8581e9f59fd1/pytorchyolo/models.py#L199
    (Ayoosh Kathuria) https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/darknet.py#L385
'''




# -------------------------------------------------------------------
# -------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np




# ------------------------------------------------------
# Used to handle weights from darknet53.conv.74 file
# Keeps pointer to know where to take data
# ------------------------------------------------------
class WeightsHandler:
    def __init__(self, src: str):
        self.ptr = 0 # pointer to data
        self.header, self.weights = self._loadWeightsFromFile(src)


    # ------------------------------------------------------
    def _loadWeightsFromFile(self, src: str):

        with open(src, 'rb') as darknet_weights:

            # first five values are header values
            header = np.fromfile(darknet_weights, dtype=np.int32, count=5)
            weights = np.fromfile(darknet_weights, dtype=np.float32)

            darknet_weights.close()

        return header, weights


    # ------------------------------------------------------
    # return torch.tensor with needed number of weights
    def getValues(self, number_of_values: int):

        weights = self.weights[self.ptr:self.ptr + number_of_values]
        self.ptr += number_of_values

        return torch.from_numpy(weights)
