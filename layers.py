import torch
from torch.distributions import distribution as dist
import torch.nn as nn
import numpy as np
import math

activation_layer = {
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'selu': nn.SELU(),
    'softmax': nn.Softmax(),
    'leaky_relu': nn.LeakyReLU()
}


class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, activation=None, batch=False):
        super(Conv2D, self).__init__()
        if activation is not None:
            self.activation = activation if type(activation) != str else activation_layer[activation]

        conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')

        self.seq = nn.Sequential(
            conv,
        )
        if batch:
            self.seq.add_module('batch', nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return self.activation(self.seq(x))


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class AttentionBlock(nn.Module):
    def __init__(self, feature, ratio):
        super(AttentionBlock, self).__init__()
        self.__build__(feature, ratio)

    def __build__(self,feature, ratio):
        self.w0 = nn.Linear(feature, feature//ratio)
        self.w1 = nn.Linear(feature//ratio, feature)
        self.g_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.g_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.compress = ChannelPool()
        self.sp_w0 = Conv2D(2, 1, 7, 1, 3, 'sigmoid', False)
        # self.sp_w0 = nn.Conv2d(2, 1, 7, stride=1, padding=3)

    def __channel_forward__(self, x):
        def __chanel_attention__(ch_input):
            temp = torch.flatten(ch_input, start_dim=1)
            temp = torch.selu(self.w0(temp))
            temp = self.w1(temp)
            return temp

        ch_avg = self.g_avg_pool(x)
        ch_avg = __chanel_attention__(ch_avg)
        ch_max = self.g_max_pool(x)
        ch_max = __chanel_attention__(ch_max)
        ch_attention = torch.sigmoid(ch_max + ch_avg)
        return ch_attention

    def __spatial_forward__(self, x):
        x_compress = self.compress(x)
        x_out = self.sp_w0(x_compress)
        return x * x_out

    def forward(self, x):
        init = x
        ch_attention = self.__channel_forward__(x)

        ch_attention = ch_attention.view((ch_attention.shape[0], ch_attention.shape[1], 1, 1))

        x = ch_attention*init
        sp_attention = self.__spatial_forward__(x)

        x = x * sp_attention
        return x


class BottleNeckBlock(nn.Module):
    def __init__(self, input_feature, attention=False, ratio=16, activation=torch.relu):
        super(BottleNeckBlock, self).__init__()
        self.input_feature = input_feature
        self.squeeze_feature = input_feature // 4
        self.attention = attention
        self.ratio = ratio
        self.activation = activation
        self.__build__()

    def __build__(self):
        self.block1 = Conv2D(self.input_feature, self.squeeze_feature, 1, 1, 0, self.activation, True)
        self.block2 = Conv2D(self.squeeze_feature, self.squeeze_feature, 3, 1, 1, self.activation, True)

        self.c3 = Conv2D(self.squeeze_feature, self.input_feature, 1, 1, 0, self.activation, False)

        if self.attention:
            self.attention = AttentionBlock(self.input_feature, self.ratio)

        self.batch1 = nn.BatchNorm2d(self.input_feature)

    def forward(self, x):
        init = x
        x = self.batch1(x)
        x = self.activation(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.c3(x)

        if self.attention:
            x = self.attention.forward(x)

        return x + init


class Hourglass(nn.Module):
    def __init__(self, feature, layers, attention=True):
        super(Hourglass, self).__init__()
        self.feature = feature
        self.layers = layers
        self.attention = attention

        self.__build__()

    def __build__(self):
        o_f = self.feature

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.up_conv = nn.ModuleList()

        for i in range(self.layers):
            self.downs.append(BottleNeckBlock(o_f, self.attention))

            self.ups.append(BottleNeckBlock(o_f, self.attention))

            self.skips.append(BottleNeckBlock(o_f, self.attention))
            if i != self.layers-1:
                self.up_conv.append(nn.ConvTranspose2d(o_f, o_f, kernel_size=2, stride=2))

        self.final_skip_1 = BottleNeckBlock(o_f, self.attention)
        self.final_skip_2 = BottleNeckBlock(o_f, self.attention)

        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)

    def forward(self, x):
        skips = []
        down = x
        # to down
        for i in range(self.layers):
            down = self.downs[i](down)
            skips.append(self.skips[i](down))
            if i != self.layers-1:
                down = self.max_pool(down)

        # middle connect
        middle = self.final_skip_1(skips[self.layers-1])
        middle = self.final_skip_2(middle)

        # to up
        for i in range(self.layers):
            if i == 0:
                up = self.ups[i](middle)
            else:
                up = self.up_conv[i-1](up) + skips[self.layers-i-1]
                up = self.ups[i](up)

        return up


class ProjLayer(nn.Module):
    def __init__(self):
        super(ProjLayer, self).__init__()

    def make_kernel(self, shape, point, radius):
        base = np.zeros(shape)

        x = math.ceil(point[0])
        y = math.ceil(point[1])

        for r in range(shape[0]):
            for c in range(shape[1]):
                base[r, c] = np.exp(-((r - y) ** 2 + (c - x) ** 2) / radius)

        return base

    def forward(self, x):
        for b in x.shape[0]:
            for j in range(21):
                x_pt = x[b,j,0].getitem()
                y_pt = x[b,j,1].getitem()
                t = self.make_kernel((64, 64), (x_pt, y_pt), 3)
        return x

class DenseBlock(nn.Module):
    def __init__(self, input_ch, growth_ch, layer, activation='relu'):
        super(DenseBlock, self).__init__()
        self.input_ch = input_ch
        self.growth_ch = growth_ch
        self.layer = layer
        self.activation = activation
        self.__build__()

    def __build__(self):
        self.block_layer = nn.ModuleList()
        for k in range(self.layer):
            input_ch = self.input_ch if k == 0 else self.input_ch + k*self.growth_ch
            output_ch = self.growth_ch
            conv1 = Conv2D(input_ch, output_ch, 1, 1, 0, self.activation, True)
            conv3 = Conv2D(output_ch, output_ch, 3, 1, 1, self.activation, True)
            seq = nn.Sequential(
                conv1,
                conv3
            )
            self.block_layer.append(seq)

    def forward(self, x):
        input_layer = x
        for block in self.block_layer:
            x = block(input_layer)
            input_layer = torch.cat([input_layer, x], 1)

        return input_layer

