import torch.nn as nn
import torchvision.ops as ops


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
