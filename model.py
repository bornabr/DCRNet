import torch
from torch import nn
import numpy as np

from utils.wrapper import ConvWrapper, LinearWrapper, MaxPoolWrapper

class DConv(nn.Module):
    def __init__(self, input_shape, dilation):
        super().__init__()
        
        self.input_shape = input_shape
        
        conv1 = ConvWrapper(self.input_shape, 2, (3, 1), dilation=dilation)
        conv2 = ConvWrapper(conv1.output_shape, 2, (1, 3), dilation=dilation)
        
        self.dconv = nn.Sequential(conv1,
                                   nn.BatchNorm2d(conv1.out_channels),
                                   nn.PReLU(num_parameters=2, init=0.3),
                                   conv2,
                                   nn.BatchNorm2d(conv2.out_channels),
                                   nn.PReLU(num_parameters=2, init=0.3))                           
        
        self.output_shape = conv2.output_shape
    
    def forward(self, x):
        
        x = self.dconv(x)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.input_shape = input_shape
        
        dconv1 = DConv(self.input_shape, 1)
        
        dconv2 = DConv(dconv1.output_shape, 2)
        
        dconv3 = DConv(dconv2.output_shape, 3)
        
        self.conv1 = nn.Sequential(dconv1, dconv2, dconv3)
        
        conv2 = ConvWrapper(self.input_shape, 2, 3, dilation=1)
        
        self.conv2 = nn.Sequential(conv2,
                                   nn.BatchNorm2d(conv2.out_channels))
        
        self.prelu1 = nn.PReLU(num_parameters=4, init=0.3)
        
        concat_shape = [self.input_shape[0], 2 * self.input_shape[1], self.input_shape[2], self.input_shape[3]]
        
        conv3 = ConvWrapper(concat_shape, 2, 1)
        
        self.conv3 = nn.Sequential(conv3, nn.BatchNorm2d(conv3.out_channels))
        
        self.identity = nn.Identity()
        
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        
        self.output_shape = conv3.output_shape
        
    def forward(self, x):
        
        identity = self.identity(x)
        
        x1 = self.conv1(x)
        
        x2 = self.conv2(x)
        
        x = torch.cat([x1, x2], dim=1)
        
        x = self.prelu1(x)
        
        x = self.conv3(x)
        
        x = x + identity
        
        x = self.prelu2(x)
        
        return x
        
        

class Encoder(nn.Module):
    def __init__(self, input_shape, reduction):
        super(Encoder, self).__init__()
        
        self.input_shape = input_shape
        
        self.input_size = np.prod(self.input_shape[1:])
        
        conv5 = ConvWrapper(self.input_shape, 2, 5)
        
        encoder_block = EncoderBlock(conv5.output_shape)
        
        self.encoder = nn.Sequential(conv5,
                                     nn.BatchNorm2d(conv5.out_channels),
                                     encoder_block)
        
        fc_input_shape = [self.input_shape[0], np.prod(encoder_block.output_shape[1:])]
        
        self.fc = LinearWrapper(fc_input_shape, self.input_size // reduction)
        
        self.output_shape = self.fc.output_shape
    
    def forward(self, x):
        
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, input_shape, expansion):
        super().__init__()
        
        self.input_shape = input_shape
        
        self.expansion = expansion
        
        out_channels = 8 * expansion
        
        groups = 4 * expansion
        
        conv1 = ConvWrapper(self.input_shape, out_channels, 3, dilation=2)
        
        conv2 = ConvWrapper(conv1.output_shape, out_channels, (1, 3), dilation=3, groups=groups)
        
        conv3 = ConvWrapper(conv2.output_shape, out_channels, (3, 1), dilation=3, groups=groups)
        
        conv4 = ConvWrapper(conv3.output_shape, 2, 3)
        
        self.conv1 = nn.Sequential(conv1,
                                   nn.BatchNorm2d(out_channels),
                                   nn.PReLU(num_parameters=out_channels, init=0.3),
                                   conv2,
                                   nn.BatchNorm2d(out_channels),
                                #    nn.ShuffleChannel(groups),
                                   nn.PReLU(num_parameters=out_channels, init=0.3),
                                   conv3,
                                   nn.BatchNorm2d(out_channels),
                                #    nn.shuffleChannel(groups),
                                   nn.PReLU(num_parameters=out_channels, init=0.3),
                                   conv4)
        
        conv1 = ConvWrapper(self.input_shape, out_channels, (1, 3))
        
        conv2 = ConvWrapper(conv1.output_shape, out_channels, (5, 1), groups=groups)
        
        conv3 = ConvWrapper(conv2.output_shape, out_channels, (1, 5), groups=groups)
        
        conv4 = ConvWrapper(conv3.output_shape, 2, (3, 1))
        
        self.conv2 = nn.Sequential(conv1,
                                   nn.BatchNorm2d(out_channels),
                                   nn.PReLU(num_parameters=out_channels, init=0.3),
                                   conv2,
                                   nn.BatchNorm2d(out_channels),
                                #    nn.ShuffleChannel(groups),
                                   nn.PReLU(num_parameters=out_channels, init=0.3),
                                   conv3,
                                   nn.BatchNorm2d(out_channels),
                                #    nn.shuffleChannel(groups),
                                   nn.PReLU(num_parameters=out_channels, init=0.3),
                                   conv4)
        
        self.identity = nn.Identity()
        
        self.prelu1 = nn.PReLU(num_parameters=4, init=0.3)
        
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        
        concat_shape = [self.input_shape[0], 2 * self.input_shape[1], self.input_shape[2], self.input_shape[3]]
        
        conv = ConvWrapper(concat_shape, 2, 1)
        
        self.conv3 = nn.Sequential(conv, nn.BatchNorm2d(conv.out_channels))
                                   
        self.output_shape = conv.output_shape
        
    def forward(self, x):
        
        identity = self.identity(x)
        
        x1 = self.conv1(x)
        
        x2 = self.conv2(x)
        
        x = torch.cat([x1, x2], dim=1)
        
        x = self.prelu1(x)
        
        x = self.conv3(x)
    
        x = x + identity
        
        x = self.prelu2(x)
        
        return x
    

class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape, expansion):
        super(Decoder, self).__init__()
        
        self.input_shape = input_shape
        
        self.output_shape = output_shape
        
        self.output_size = np.prod(self.output_shape[1:])
        
        self.fc = LinearWrapper(self.input_shape, self.output_size)
        
        conv5 = ConvWrapper(self.output_shape, 2, 5)
        
        decoderBlock1 = DecoderBlock(conv5.output_shape, expansion)
        
        decoderBlock2 = DecoderBlock(decoderBlock1.output_shape, expansion)
        
        self.decoder = nn.Sequential(conv5,
                                     nn.BatchNorm2d(conv5.out_channels),
                                     nn.PReLU(num_parameters=2, init=0.3),
                                     decoderBlock1,
                                     decoderBlock2)
        
        
        self.output_shape = decoderBlock2.output_shape
    
    def forward(self, x):
        
        x = self.fc(x)
        
        x = x.view(x.shape[0], *self.output_shape[1:])
        
        x = self.decoder(x)
        
        return x

class DCRNet(nn.Module):
    def __init__(self, input_shape, reduction=4, expansion=1):
        super(DCRNet, self).__init__()
        
        self.input_shape = input_shape
        
        encoder = Encoder(self.input_shape, reduction)
        
        decoder = Decoder(encoder.output_shape, self.input_shape, expansion)
        
        self.net = nn.Sequential(encoder, decoder, nn.Sigmoid())
        
        # TODO: Add kaiming init
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.net(x)
        
        