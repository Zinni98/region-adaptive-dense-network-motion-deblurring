import torch
from torch import nn
from einops import rearrange
from torch import einsum


class SelfAttention(nn.Module):
    """
    Self attention module inspired by
    "Self Attention in Generative Adversarial Networks":
    https://arxiv.org/pdf/1805.08318.pdf
    """
    def __init__(self, in_channels: int):
        super(SelfAttention, self).__init__()
        # Naming is kept consistent with the paper:
        # https://arxiv.org/pdf/1805.08318.pdf
        self.f = nn.Conv2d(in_channels=in_channels,
                           out_channels=in_channels//8,
                           kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channels,
                           out_channels=in_channels//8,
                           kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channels,
                           out_channels=in_channels//8,
                           kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_channels//8,
                           out_channels=in_channels,
                           kernel_size=1)
        # self.gamma = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x):
        f_of_x = self.f(x)
        g_of_x = self.g(x)
        h_of_x = self.h(x)

        # Note that f_of_x has been flattened and transposed
        f_of_x = rearrange(f_of_x, "b c h w -> b (h w) c")
        # (b, n, c~) c~ being c//8

        g_of_x = rearrange(g_of_x, "b c h w -> b c (h w)")
        h_of_x = rearrange(h_of_x, "b c h w -> b c (h w)")
        # (b, c~, n)

        # Softmax done column-wise (see paper)
        attention_map = nn.Softmax(einsum("bnc,bcn->bnn", f_of_x, g_of_x),
                                   dim=1)

        attention_out_flat = einsum("bcn,bnn->bcn", h_of_x, attention_map)
        attention_out = self.v(rearrange("b c (h w) -> b c h w", attention_out_flat))
        # (b c h w)

        out = torch.add(x, attention_out)

        return out
