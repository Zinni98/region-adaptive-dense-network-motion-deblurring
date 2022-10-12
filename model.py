from ast import If
import torchvision.ops as ops
from torch import nn
import torch


# TODO: write the version 2 of deformable convolution and compare the two
class DenseDeformableModule(nn.Module):
    """
    Performs a deformable convolution as it was a regular convolution
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DenseDeformableModule, self).__init__()
        self.padding = padding
        self.offset_convolution = nn.Conv2d(in_channels=in_channels,
                                            out_channels=2 * (kernel_size**2),
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            bias=bias)
        self.standard_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       padding=self.padding,
                                       stride=stride,
                                       bias=bias)
        nn.init.xavier_uniform(self.offset_convolution)
        nn.init.xavier_uniform(self.standard_conv)

    def forward(self, x):
        offset = self.offset_convolution(x)

        x = ops.deform_conv2d(x,
                              offset=offset,
                              weight=self.standard_conv.weight,
                              bias=self.standard_conv.bias,
                              padding=self.padding)


class SpaceToDepthModule(nn.Module):
    def __init__(self, block_size=None):
        super(SpaceToDepthModule, self).__init__()
        self.block_size = block_size

    # Inspired by https://stackoverflow.com/questions/58857720/is-there-an-equivalent-pytorch-function-for-tf-nn-space-to-depth
    def forward(self, x):
        sz = x.size()
        if len(sz) == 3:
            raise ValueError("Input should be batched, if you are trying to \
                              use the module only for one instance, perform \
                              the following operation before feeding the \
                              input to this module: \n  \
                              input = input.unsqueeze(dim = 0)")
        elif len(sz) != 4:
            raise ValueError(f"Input tensor number of dimensions is not correct\n \
                              expected: 4 \n \
                              given: {sz}")

        batch_sz, n_channels, height, width = sz
        return x.view(batch_sz,
                      n_channels * self.block_size ** 2,
                      height // self.block_size,
                      width // self.block_size)


class RegionAdaptiveNetwork(nn.Module):
    def __init__(self):
        super(RegionAdaptiveNetwork, self).__init__()


if __name__ == "__main__":
    a = torch.tensor([[[
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                    ]]]).float()
    sptd = SpaceToDepthModule(block_size=2)
    print(a.size())
    print(sptd(a).size())
