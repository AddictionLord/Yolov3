import torch
import torch.nn as nn
import numpy as np

from weights_handler import WeightsHandler




# ------------------------------------------------------
# 
# ------------------------------------------------------
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, **kwargs):
        super(CNNBlock, self).__init__()

        self.batch_norm = batch_norm
        if batch_norm:
            # If batch norm is performed, leaky relu activation fcn is used
            # In this case bias=False -> no bias to learn/load
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1)
            )
            
        else:
            # Without batch norm we only use linear activation fcn
            # -> output is same as input (no change is applied)
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, **kwargs)
            )


    # ------------------------------------------------------
    def forward(self, x):

        return self.block(x)


    # ------------------------------------------------------
    # Loads weights to whole CNNBlock from np.ndarray data
    # loaded from darknet53.conv.74 file
    def loadWeights(self, weights: WeightsHandler):

        for layer in self.block:

            # TODO: Create two private fcns to load weights to Conv2d and BatchNorm
            # to make this fcn cleaner, two fcns are just for inside use
            if isinstance(layer, nn.Conv2d):

                num_of_weights = layer.weight.numel()
                cnn_weights = weights.getValues(num_of_weights)
                layer.weight.data.copy_(cnn_weights.view_as(layer.weight.data))

                if not self.batch_norm:
                
                    num_of_biases = layer.bias.numel()
                    cnn_biases = weights.getValues(num_of_biases)
                    layer.weight.data.copy_(cnn_biases.view_as(layer.bias.data))

            if isinstance(layer, nn.BatchNorm2d):

                # number of params are same for biases, weights, means and vars
                num_of_parameters = layer.bias.numel()

                bn_biases = weights.getValues(num_of_parameters)
                layer.bias.data.copy_(bn_biases.view_as(layer.bias.data))

                bn_weights = weights.getValues(num_of_parameters)
                layer.weight.data.copy_(bn_weights.view_as(layer.weight.data))

                bn_run_means = weights.getValues(num_of_parameters)
                layer.running_mean.data.copy_(bn_run_means.view_as(layer.running_mean.data))

                bn_run_vars = weights.getValues(num_of_parameters)
                layer.running_var.data.copy_(bn_run_vars.view_as(layer.running_var.data))




# ------------------------------------------------------  
#   
# ------------------------------------------------------    
class ResidualBlock(nn.Module):
    def __init__(self, num_of_repeats, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential()
        res_block = nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=1, padding=0),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1)
        )

        for repeat in range(num_of_repeats):

            self.block = nn.Sequential(*self.block, *res_block)


    # ------------------------------------------------------
    def forward(self, x):

        return self.block(x)






# ------------------------------------------------------
# Main - mostly for testing purposes
# ------------------------------------------------------
if __name__ == '__main__':

    # t = torch.rand(1, 3, 256, 256)
    # kernel_size = 3
    # padding = 1 if kernel_size == 3 else 0

    # pre = nn.Sequential(
    #     CNNBlock(3, 32, 3, stride=1, padding=padding),
    #     CNNBlock(32, 64, 3, stride=2, padding=padding)
    # )
    # out = pre(t)
    # print(out.shape)

    # res = ResidualBlock(1, 64)
    # out = res(out)
    # print(out.shape)    

    from load import weights
    t = torch.rand(1, 3, 256, 256)
    cnn = CNNBlock(3, 32, kernel_size=3, stride=1, padding=1)

    cnn.loadWeights(weights)