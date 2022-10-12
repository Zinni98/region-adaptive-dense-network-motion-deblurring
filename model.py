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
    def __init__(self, scale_factor=2):
        super(SpaceToDepthModule, self).__init__()
        self.shuff = nn.PixelUnshuffle(2)

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
        return self.shuff(x)


class RegionAdaptiveNetwork(nn.Module):
    def __init__(self):
        super(RegionAdaptiveNetwork, self).__init__()


if __name__ == "__main__":
    val = 8
    a = torch.tensor([[[
                        [i for i in range(((j // val) * val) + 1, j + val)]
                        for j in range(1, val*val + 1, val)
                    ]]]).float()
    print(a)
