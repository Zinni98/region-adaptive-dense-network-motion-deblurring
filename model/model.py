from torch import nn
import torch
from torchvision.models import DenseNet121_Weights
from space_to_depth import SpaceToDepthModule


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
