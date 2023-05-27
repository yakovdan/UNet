import itertools

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, num_layers=2):
        super().__init__()
        self.conv_units = nn.ModuleList([self.build_conv_unit(in_channels, out_channels, use_batchnorm)] +
                                        [self.build_conv_unit(out_channels, out_channels, use_batchnorm) for i in
                                         range(num_layers - 1)])
        self.init_conv_unit_weights()

    @staticmethod
    def build_conv_unit(in_channels, out_channels, use_batchnorm=True):
        if use_batchnorm:
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", bias=False),
                                 nn.BatchNorm2d(out_channels), nn.ReLU())
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", bias=True),
                             nn.ReLU())

    def init_conv_unit_weights(self):
        for unit in self.conv_units:
            nn.init.xavier_uniform_(unit[0].weight)

    def forward(self, x):
        for unit in self.conv_units:
            x = unit(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, use_batchnorm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_transpose_conv=True, use_batchnorm=True):
        super().__init__()
        if use_transpose_conv:
            self.upscale_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            nn.init.xavier_uniform_(self.upscale_layer.weight)
            self.conv_block = ConvBlock(skip_channels + out_channels, out_channels, use_batchnorm, 2)
        else:
            self.upscale_layer = nn.Upsample(mode="bilinear", scale_factor=2)
            self.conv_block = ConvBlock(skip_channels + in_channels, out_channels, use_batchnorm, 3)

    def forward(self, x, skip):
        x = self.upscale_layer(x)
        x = torch.cat((skip, x), dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, channels_list, use_transpose_conv=True, use_batchnorm=True):
        super().__init__()
        self.down_path = nn.ModuleList()
        up_path = []
        for in_chan, out_chan in zip(channels_list[:-1], channels_list[1:]):
            self.down_path.append(DownBlock(in_chan, out_chan,use_batchnorm))
            up_path.append(UpBlock(2*out_chan, out_chan, out_chan, use_transpose_conv, use_batchnorm))

        self.up_path = nn.ModuleList(reversed(up_path))
        cross_path_width = 2*channels_list[-1]
        self.cross_path_block = ConvBlock(channels_list[-1], cross_path_width, use_batchnorm)
        self.output_conv = nn.Conv2d(channels_list[1], 1, 1, padding="same", bias=True)

    def forward(self, x):
        output_List = [(x, )]
        for block in self.down_path:
            output_List.append(block(output_List[-1][0]))
        downpath_output = output_List[-1][0]
        cross_output = self.cross_path_block(downpath_output)
        for idx, block in enumerate(self.up_path):
            cross_output = block(cross_output, output_List[-1-idx][1])
        output = self.output_conv(cross_output)
        return output


def test():
    print('testing')
    unet = UNet([3, 64, 128, 256, 512]).to('cuda:0')
    test_input = torch.randn((1, 3, 1280, 1920)).to('cuda:0')
    out_prediction = unet(test_input)
    assert(out_prediction.shape[2:] == test_input.shape[2:])
    print("Pass")

if __name__ == "__main__":
    test()
