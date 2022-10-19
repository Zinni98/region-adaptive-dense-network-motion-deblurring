import torch
import torch.nn as nn


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


if __name__ == "__main__":
    x = torch.randn((2, 3, 32, 32))
    net = SpaceToDepthModule()
    print(net(x).shape)
