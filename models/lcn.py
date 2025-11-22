import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoc1d(nn.Module):

    def __init__(
        self, input_dim, in_channels, out_channels, filter_size, stride=1, bias=False
    ):
        """
        Args:
            input_dim: The spatial dimension of the input
            in_channels: The number of input channels
            out_channels: The number of output channels
            filter_size: The size of the local kernel
            stride: The stride (kernel applied every stride pixels)
            bias: True for adding bias
        """
        super().__init__()

        self.input_dim = input_dim
        self.filter_size = filter_size
        self.stride = stride
        self.num_patches = (input_dim-filter_size)//stride + 1
        self.filter = nn.Parameter(
            torch.randn( out_channels, in_channels, self.num_patches, filter_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, self.num_patches))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).
        
        Returns:
            The convolution of x with self.filter, tensor of size (batch_size, out_channels, num_patches),
            num_patches = (input_dim-filter_size)//stride+1
        """
        out = F.unfold( # break input in patches [bs, in_channels, num_patches, filter_size]
            x.unsqueeze(-1),
            kernel_size=(self.filter_size,1),
            dilation=1,
            padding=0,
            stride=self.stride,
        ).reshape(-1, self.filter.size(1), self.filter.size(3), self.filter.size(2)).transpose(-1,-2)

        out = out[:, None] * self.filter # [bs, out_channels, in_channels, num_patches, filter_size]
        out = out.sum(dim=[-1,-3]) *(self.filter.size(1)*self.filter.size(3))**-.5  # [bs, out_channels, num_patches]
        if self.bias is not None:
            out += self.bias

        return out


class hLCN(nn.Module):
    def __init__(
        self, input_dim, patch_size, in_channels, nn_dim, out_channels, num_layers, bias=False, norm='std'
    ):
        """
        Hierarchical LCN

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

        receptive_field = patch_size**num_layers
        assert input_dim % receptive_field == 0, 'patch_size**num_layers must divide input_dim!'

        self.hidden = nn.Sequential(
            nn.Sequential(
                MyLoc1d(
                    input_dim, in_channels, nn_dim, patch_size, stride=patch_size, bias=bias
                ),
                nn.ReLU(),
            ),
            *[nn.Sequential(
                    MyLoc1d(
                        input_dim//(patch_size**l), nn_dim, nn_dim, patch_size, stride=patch_size, bias=bias
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
            Output of a hierarchical LCN, tensor of size (batch_size, out_dim)
        """
        x = self.hidden(x)
        x = x.mean(dim=[-1]) # Global Average Pooling if the final spatial dimension is > 1
        x = x @ self.readout / self.norm
        return x
