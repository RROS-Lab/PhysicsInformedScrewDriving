# This code is taken directly from the multi_modal representation reporsoitory, Only minor updates are made:
# https://github.com/stanford-iprl-lab/multimodal_representation/blob/master/multimodal/models/base_models/layers.py
# 
# #

import torch


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



class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



class GRUNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = torch.nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("ccda")
        return hidden