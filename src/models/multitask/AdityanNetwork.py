import torch
from torch import nn
from collections import OrderedDict


def conv1x1(in_channels, out_channels):
    """ 2D convolution which uses a kernel size of 1"""

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    """ 2D convolution which uses a kernel size of 3"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride,
                     padding=dilation, dilation=dilation)


class ConvReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            OrderedDict(
                [
                    ('Conv', conv3x3(in_channels, out_channels)),
                    ('ReLU', nn.ReLU())
                ]
            )
        )


class LevelBlock(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__(
            OrderedDict(
                [
                    ('ConvRelu1', ConvReLU(in_channels, mid_channels)),
                    ('ConvRelu2', ConvReLU(mid_channels, out_channels))
                ])
        )


class Conv3x3Sigmoid(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(
            OrderedDict(
                [
                    ('Conv', conv3x3(in_channels, in_channels)),
                    ('Sigmoid', nn.Sigmoid())
                ]
            )
        )


class Adityan(nn.Module):

    name = "Adityan network"

    def __init__(self, sequences, regions, width):
        super(Adityan, self).__init__()

        widths = [width * 2 ** i for i in range(5)]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0], widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1], widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2], widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3], widths[3])

        # Bottleneck
        self.bottleneck = LevelBlock(widths[3], widths[4], widths[3])

        # Decoders
        self.decoder4 = LevelBlock(widths[3] * 2, widths[3], widths[2])
        self.decoder3 = LevelBlock(widths[2] * 2, widths[2], widths[1])
        self.decoder2 = LevelBlock(widths[1] * 2, widths[1], widths[0])

        # Upsample and downsample layers
        self.upsample4 = nn.ConvTranspose2d(in_channels=widths[3], out_channels=widths[3], kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=widths[2], out_channels=widths[2], kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels=widths[1], out_channels=widths[1], kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels=widths[0], out_channels=widths[0], kernel_size=2, stride=2)
        self.downsample = nn.MaxPool2d(2, 2)

        # maps
        self.segmap = LevelBlock(widths[0] * 2, widths[0], widths[0])
        self.recmap = LevelBlock(widths[0] * 2, widths[0], widths[0])
        self.classmap = nn.Sequential(
            self.downsample,
            self.downsample,
            self.downsample,
            ConvReLU(widths[0] * 2, 32),
            nn.AvgPool2d(kernel_size=16),
            nn.Flatten(),
            nn.Linear(32, 1000),
            nn.ReLU(),
            nn.Linear(1000, 3)
        )

        # Output
        self.seg_out = conv1x1(widths[0], regions)
        self.rec_out = conv3x3(widths[0], regions)

    def forward(self, x):

        # Encoding phase
        e1 = self.encoder1(x)
        p1 = self.downsample(e1)
        e2 = self.encoder2(p1)
        p2 = self.downsample(e2)
        e3 = self.encoder3(p2)
        p3 = self.downsample(e3)
        e4 = self.encoder4(p3)
        p4 = self.downsample(e4)

        # Bottleneck phase
        bottleneck = self.bottleneck(p4)

        # Decoding phase + skip connections
        up4 = self.upsample4(bottleneck)
        d4 = self.decoder4(torch.cat([e4, up4], dim=1))
        up3 = self.upsample3(d4)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample2(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample1(d2)
        d1 = torch.cat([e1, up1], dim=1)

        # seg map
        segmap = self.segmap(d1)
        seg_out = self.seg_out(segmap)
        #
        # # rec map
        recmap = self.recmap(d1)
        rec_out = nn.functional.sigmoid(self.rec_out(recmap))

        # cls map
        cls_map = self.classmap(d1)

        # Output
        return [cls_map, rec_out, seg_out]


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    seq_ouput = torch.rand(1, 1, 128, 128)

    model = Adityan(sequences=1, regions=1, width=64)
    labels, rec, seg = model(seq_input)
    print(labels)

