import torch
import torch.nn as nn




# ------------------------------------------------------
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        

    def forward(self, x):

        return self.block(x)




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


    def forward(self, x):

        return self.block(x)






if __name__ == '__main__':

    t = torch.rand(1, 3, 256, 256)
    kernel_size = 3
    padding = 1 if kernel_size == 3 else 0

    pre = nn.Sequential(
        CNNBlock(3, 32, 3, stride=1, padding=padding),
        CNNBlock(32, 64, 3, stride=2, padding=padding)
    )
    out = pre(t)
    print(out.shape)

    res = ResidualBlock(1, 64)
    out = res(out)
    print(out.shape)    