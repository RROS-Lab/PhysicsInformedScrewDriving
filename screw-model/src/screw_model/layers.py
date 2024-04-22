# This code is taken directly from the multi_modal representation reporsoitory, Only minor updates are made:
# https://github.com/stanford-iprl-lab/multimodal_representation/blob/master/multimodal/models/base_models/layers.py
# 
# #

import torch
from sindy_utils import *


class CausalConv1D(torch.nn.Conv1d):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation = 1, bias=True, device="cuda"):
        
        self._padding = (kernel_size - 1)* dilation
        self.device = device
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding = self._padding, dilation=dilation, bias=bias,device=device)


    
    def forward(self, input):
        output = super().forward(input.to(self.device))

        if(self._padding != 0):
            return output[:,:,:-self._padding]

        return output


class CausalConvTransposed1D(torch.nn.ConvTranspose1d):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation = 1, bias=True, device="cuda"):
        
        self._padding = (kernel_size - 1)* dilation
        self.device = device
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding = self._padding, dilation=dilation, bias=bias,device=device)


    
    def forward(self, input):
        output = super().forward(input.to(self.device))

        if(self._padding != 0):
            return output[:,:,:-self._padding]

        return output


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True, device="cpu"):
    """`same` convolution with LeakyReLU, i.e. output shape equals input shape.
  Args:
    in_planes (int): The number of input feature maps.
    out_planes (int): The number of output feature maps.
    kernel_size (int): The filter size.
    dilation (int): The filter dilation factor.
    stride (int): The filter stride.
  """
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=same_padding,
            dilation=dilation,
            bias=bias,
            device=device
        ),
        torch.nn.LeakyReLU(0.1, inplace=True).to(device),
    )



class Residual(torch.nn.Module):

    def __init__(self, channels, device="cuda"):
        super().__init__()

        self.conv1 = conv2d(channels, channels, bias=False)
        self.conv2 = conv2d(channels, channels, bias=False)
        self.batch_norm1 = torch.nn.BatchNorm2d(channels)
        self.batch_norm2 = torch.nn.BatchNorm2d(channels)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    
    def forward(self, x):
        x = x.to("cuda")
        output = self.activation(x)
        output = self.activation(self.batch_norm1(self.conv1(output)))
        output = self.batch_norm2(self.conv2(output))
        
        return output+x

class Flatten(torch.nn.Module):
    """Flattens convolutional feature maps for fc layers.
  """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)

