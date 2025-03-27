import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv1d(nn.Module):

    def __init__(
        self, in_channels, out_channels, filter_size, stride=1, bias=False
    ):
        """
        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            filter_size: The size of the convolutional kernel
            stride: The stride (conv. ker. applied every stride pixels)
            bias: True for adding bias
        """
        super().__init__()

        self.filter_size = filter_size
        self.stride = stride
        self.filter = nn.Parameter( torch.randn( out_channels, in_channels, filter_size))
        if bias:
            self.bias = nn.Parameter( torch.randn( out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).
        
        Returns:
            The convolution of x with self.filter, tensor of size (batch_size, out_channels, out_dim),
            out_dim = (input_dim-filter_size)//stride+1
        """

        return F.conv1d(x, self.filter, self.bias, stride=self.stride) / (self.filter.size(1)*self.filter.size(2))**.5


class hCNN(nn.Module):
    def __init__(
        self, input_dim, patch_size, in_channels, nn_dim, out_channels, num_layers, bias=False, norm='std'
    ):
        """
        Hierarchical CNN

        Args:
            input_dim: The input dimension.
            patch_size: The size of the patches.
            in_channels: The number of input channels.
            nn_dim: The number of hidden neurons per layer.
            out_channels: The output dimension.
            num_layers: The number of layers.
            bias: True for adding bias.
            norm: Scaling factor for the readout layer.
        """
        super().__init__()

        self.num_layers = num_layers

        receptive_field = patch_size**num_layers
        assert input_dim % receptive_field == 0, 'patch_size**num_layers must divide input_dim!'

        self.hidden = nn.Sequential(
            nn.Sequential(
                MyConv1d(
                    in_channels, nn_dim, patch_size, stride=patch_size, bias=bias
                ),
                nn.ReLU(),
            ),
            *[nn.Sequential(
                    MyConv1d(
                        nn_dim, nn_dim, patch_size, stride=patch_size, bias=bias
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        self.readout = nn.Parameter(
            torch.randn(nn_dim, out_channels)
        )
        if norm=='std':
            self.norm = nn_dim**.5 # standard NTK scaling
        elif norm=='mf':
            self.norm = nn_dim # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).
        
        Returns:
            Output of a hierarchical CNN, tensor of size (batch_size, out_dim)
        """
        x = self.hidden(x)
        x = x.mean(dim=[-1]) # Global Average Pooling if the final spatial dimension is > 1
        x = x @ self.readout / self.norm
        return x
