import torchvision.ops as ops
from torch import nn
import torch
from torchvision.models import DenseNet121_Weights


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
        # initialize weights to a uniform distribution
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
        self.shuff = nn.PixelUnshuffle(scale_factor)

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
        densenet = torch.hub.load("pytorch/vision:v0.10.0",
                                  "densenet121",
                                  weights=DenseNet121_Weights.DEFAULT)
        fe_densenet = list(densenet.children())[0]
        layers_densenet = dict(fe_densenet.named_children())
        self.sptd = SpaceToDepthModule()
        # 12 dense layers 128 in channels, 512 out
        self.denseblock1 = layers_densenet["denseblock2"]
        # 16 dense layers 512 in channels, 1024 out channels
        self.denseblock2 = layers_densenet["denseblock4"]
        # 24 dense layers 256 in channels, 1024 out channels
        self.denseblock3 = layers_densenet["denseblock3"]


if __name__ == "__main__":
    densenet = torch.hub.load("pytorch/vision:v0.10.0",
                              "densenet121",
                              weights=DenseNet121_Weights.DEFAULT)
    fe = list(densenet.children())[0]
    print(fe)
    # print(dict(fe.named_children())["denseblock3"])
    """val = 12
    a = torch.tensor([[[
                        [i for i in range(((j // val) * val) + 1, j + val)]
                        for j in range(1, val*val + 1, val)
                    ]]]).float()
    sptd = SpaceToDepthModule(2)
    print(a)
    print(sptd(a))
"""
