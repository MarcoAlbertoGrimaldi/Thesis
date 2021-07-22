import torch
import torch.nn.functional as F
from torch import nn


class SoftConv2d(nn.Module):
    """
    Apply classical conv2d on input tensor and weights output using the result of another conv2d applied
    to a soft mask
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_type='zero',
                 padding=0,
                 dilation=1,
                 bias=True,
                 norm=None,
                 activation="lrelu"):
        super().__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, f"Unsupported padding type: {pad_type}"

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm is None:
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, f"Unsupported activation: {activation}"

        self.dilation = dilation
        self.bias = bias
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, bias=bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding=0, dilation=dilation, bias=False)

        nn.init.constant_(self.mask_conv.weight, 1)
        nn.init.kaiming_normal_(self.input_conv.weight)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        """
        Forward pass: input is masked before going into convolution operation, mask is update to reflect
        the changes performed by the conv2d
        """

        x = self.pad(x)
        with torch.no_grad():
            mask = self.pad(mask)
            out_mask = self.mask_conv(mask)

        output = self.input_conv(x * mask)

        if self.input_conv.bias is not None:
            out_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            out_bias = torch.zeros_like(output)

        # mask_sum is the sum of the binary mask at every partial convolution location
        print(out_mask)
        mask_less_then_one = (out_mask <= 1)
        print(mask_less_then_one)
        # temporarily sets less_then_one values to one to ease output calculation
        mask_sum = out_mask.masked_fill_(mask_less_then_one, 1.0)

        # output at each location as follows:
        # output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
        # output = 0 ; if M_sum < 1
        output = (output - out_bias) / mask_sum + out_bias
        output = output.masked_fill_(mask_less_then_one, 0.0)

        # mask is updated at each location
        new_mask = out_mask / (self.mask_conv.kernel_size[0] * self.mask_conv.kernel_size[1])

        if self.norm:
            output = self.norm(output)

        if self.activation:
            output = self.activation(output)

        return output, new_mask

