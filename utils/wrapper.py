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