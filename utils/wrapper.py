import math

import torch
from torch import nn

import numpy as np

def num2tuple(num):
        return num if isinstance(num, tuple) else (num, num)

class ConvWrapper(nn.Conv2d):
    def __init__(self, input_shape, *args, **kwargs):
        self.input_shape = tuple(input_shape)
        in_channels = self.input_shape[1]
        
        kernel_size = args[1]
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        if 'dilation' in kwargs:
            dilation = kwargs['dilation']
        else:
            dilation = 1
        
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        if 'padding' in kwargs:
            padding = kwargs['padding']
        else:
            padding = tuple([((kernel_size[i] + (kernel_size[i] - 1) * (dilation[i] - 1)) - 1) // 2 for i in range(len(kernel_size))])
        
        super(ConvWrapper, self).__init__(in_channels, *args, padding=padding, **kwargs)
        self.output_shape = self.calculate_output_shape()

    def calculate_output_shape(self):
        input_dim, kernel_size, stride, pad, dilation = num2tuple(self.input_shape[2:]), num2tuple(self.kernel_size), num2tuple(self.stride), num2tuple(self.padding), num2tuple(self.dilation)
        pad = num2tuple(pad[0]), num2tuple(pad[1])
        
        output_shape = []
        for i in range(len(input_dim)):
            output_shape.append(math.floor((input_dim[i] + sum(pad[i]) - dilation[i]*(kernel_size[i]-1) - 1) / stride[i] + 1))
   
        return self.input_shape[:1] + (self.out_channels, ) +  tuple(output_shape)

class ConvTransWrapper(nn.Module):
    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(ConvTransWrapper, self).__init__()
        self.in_channels = input_shape[1]
        self.input_shape = tuple(input_shape)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        if padding_mode == 'same':
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        elif padding_mode == 'valid':
            self.padding = (0, 0)
        else:
            self.padding = padding
        
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        
        self.convTranspose = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding, self.groups, self.bias, self.dilation, self.padding_mode)
        self.output_shape = self.calculate_output_shape()

    def forward(self, input):   
        return self.convTranspose(input)
  
    def calculate_output_shape(self):
        input_dim, kernel_size, stride, pad, dilation, out_pad = num2tuple(self.input_shape[2:]), num2tuple(self.kernel_size), num2tuple(self.stride), num2tuple(self.padding), num2tuple(self.dilation), num2tuple(self.output_padding)
        pad = num2tuple(pad[0]), num2tuple(pad[1])
        
        output_shape = []		
        for i in range(len(input_dim)):
            output_shape.append((input_dim[i] - 1)*stride[i] - sum(pad[i]) + dilation[i]*(kernel_size[i]-1) + out_pad[i] + 1)

        return self.input_shape[:1] + (self.out_channels, ) + tuple(output_shape)

class MaxPoolWrapper(nn.Module):
    def __init__(self, input_shape, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPoolWrapper, self).__init__()
        self.input_shape = tuple(input_shape)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.maxPool = nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices, self.ceil_mode)
        self.output_shape = self.calculate_output_shape()

    def forward(self, input):
        return self.maxPool(input)
  
    def calculate_output_shape(self):
        input_dim, kernel_size, stride, pad, dilation = num2tuple(self.input_shape[2:]), num2tuple(self.kernel_size), num2tuple(self.stride), num2tuple(self.padding), num2tuple(self.dilation)
        pad = num2tuple(pad[0]), num2tuple(pad[1])
        output_shape = []
        for i in range(len(input_dim)):
            output_shape.append(math.floor((input_dim[i] + sum(pad[i]) - dilation[i] * (kernel_size[i] -1) -1 ) / stride[i] ) + 1)
        
        return self.input_shape[:2] + tuple(output_shape)

class MaxUnpoolWrapper(nn.Module):
    def __init__(self, input_shape, kernel_size, stride=None, padding=0, output_shape=None):
        super(MaxUnpoolWrapper, self).__init__()
        self.input_shape = tuple(input_shape)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxunPool = nn.MaxUnpool2d(self.kernel_size, self.stride, self.padding)
        self.output_shape = output_shape if output_shape is not None else self.calculate_output_shape()

    def forward(self, input, indices, output_shape=None):
        if output_shape is not None:
            self.output_shape = output_shape
        return self.maxunPool(input, indices, output_shape=self.output_shape)
  
    def calculate_output_shape(self):
        input_dim, kernel_size, stride, pad = num2tuple(self.input_shape[2:]), num2tuple(self.kernel_size), num2tuple(self.stride), num2tuple(self.padding)
        pad = num2tuple(pad[0]), num2tuple(pad[1])
        output_shape = []
        for i in range(len(input_dim)):
            output_shape.append((input_dim[i] - 1) * stride[i] - sum(pad[i]) + kernel_size[i])
        return self.input_shape[:2] + tuple(output_shape)

class LinearWrapper(nn.Module):
    def __init__(self, input_shape, out_features, bias=True):
        super(LinearWrapper, self).__init__()
        
        self.input_shape = tuple(input_shape)
        
        if isinstance(self.input_shape, int):
            self.in_features = self.input_shape
        else:
            self.in_features = math.prod(input_shape[1:])
        
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)
        self.output_shape = self.calculate_output_shape()

    def forward(self, input):
        return self.linear(input)
  
    def calculate_output_shape(self):
        if isinstance(self.input_shape, int):
            return (self.input_shape[0], self.out_features)
        else:
            return (-1, self.out_features)