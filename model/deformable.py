import torch.nn as nn
import torchvision.ops as ops
import torch


# TODO: write the version 2 of deformable convolution and compare the two
class DeformUnit(nn.Module):
    """
    Performs a deformable convolution as it was a regular convolution
    """
    def __init__(self,
                 in_channels,
                 growth_rate,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformUnit, self).__init__()
        self.padding = padding
        self.offset_convolution = nn.Conv2d(in_channels=in_channels,
                                            out_channels=2 * (kernel_size**2),
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=self.padding,
                                            bias=bias)
        self.standard_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=growth_rate,
                                       kernel_size=kernel_size,
                                       padding=self.padding,
                                       stride=stride,
                                       bias=bias)

    def _initialize_model(self):
        # initialize weights to a uniform distribution
        nn.init.xavier_uniform(self.offset_convolution)
        nn.init.xavier_uniform(self.standard_conv)

    def forward(self, x):
        offset = self.offset_convolution(x)

        out = ops.deform_conv2d(x,
                                offset=offset,
                                weight=self.standard_conv.weight,
                                bias=self.standard_conv.bias,
                                padding=self.padding)
        out = torch.cat((x, out), dim=1)
        return out


class DenseDeformableModule(nn.Module):
    """
    out_channels = (in_channels+growth_rate*6)*compress_rate+in_channels
    """
    def __init__(self,
                 in_channels,
                 growth_rate=32,
                 compress_rate=.5):
        self.gr = growth_rate
        self.out_ch = in_channels+self.gr*6
        self.deform = nn.Sequential(self._get_std_deform_unit(in_channels),
                                    self._get_std_deform_unit(in_channels+self.gr),
                                    self._get_std_deform_unit(in_channels+self.gr*2),
                                    self._get_std_deform_unit(in_channels+self.gr*3),
                                    self._get_std_deform_unit(in_channels+self.gr*4),
                                    self._get_std_deform_unit(in_channels+self.gr*5),
                                    nn.Conv2d(self.out_ch,
                                              self.out_ch * compress_rate,
                                              1)
                                    )

    def _get_std_deform_unit(self, in_ch):
        return DeformUnit(in_ch,
                          self.gr,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)

    def forward(self, x):
        out = self.deform(x)
        out = torch.cat((x, out), dim=1)
        return out


if __name__ == "__main__":
    x = torch.randn((2, 3, 8, 8))
    model = DeformUnit(3, 6)
    y = model(x)
    print(y.shape)
