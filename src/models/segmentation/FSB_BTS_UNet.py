import torch
from torch import nn
from collections import OrderedDict


def conv1x1(in_channels, out_channels):
    """ 2D convolution which uses a kernel size of 1"""

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=False):
    """ 2D convolution which uses a kernel size of 3"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv5x5(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=False):
    """ 2D convolution which uses a kernel size of 3"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

class ConvInNormLeReLU(nn.Sequential):
    """
    This class stacks a 2D Convolution, Instance Normalization and Leaky ReLU layers.

    Params
    ******
        - in_channels: Number of input channels
        - out_channels: Number of output channels

    """

    def __init__(self, in_channels, out_channels):
        super(ConvInNormLeReLU, self).__init__(
            OrderedDict(
                [
                    ('Conv', conv3x3(in_channels, out_channels)),
                    ('InNorm', nn.InstanceNorm2d(out_channels)),
                    ('LeReLU', nn.LeakyReLU(inplace=True))
                ]
            )
        )


class LevelBlock(nn.Sequential):
    """
    This class stacks two blocks of ConvInNormLeReLU (2D Convolution, Instance Normalization and Leaky ReLU layers).

    Params
    ******
        - in_channels: Number of input channels
        - mid_channels: Number of channels between the first and the second block
        - out_channels: Number of output channels

    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(LevelBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvInNormLRelu1', ConvInNormLeReLU(in_channels, mid_channels)),
                    ('ConvInNormLRelu2', ConvInNormLeReLU(mid_channels, out_channels))
                ])
        )


class FSB_BTS_UNet(nn.Module):
    """
    This class implements a variation of 2D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "Full-Scale-Bridge BTS U-Net"

    def __init__(self, sequences, regions, width, deep_supervision):
        super(FSB_BTS_UNet, self).__init__()

        self.deep_supervision = deep_supervision
        widths = [width * 2 ** i for i in range(4)]

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0] // 2, widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1] // 2, widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2] // 2, widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3] // 2, widths[3])

        # Bottleneck
        self.bottleneck = LevelBlock(widths[3], widths[3], widths[3])
        self.bottleneck2 = ConvInNormLeReLU(widths[3] * 2, widths[2])

        # Decoders
        self.decoder3 = LevelBlock(widths[2] * 2, widths[2], widths[1])
        self.decoder2 = LevelBlock(widths[1] * 2, widths[1], widths[0])
        self.decoder1 = LevelBlock(widths[0] * 2 + widths[-1], widths[0], widths[0] // 2)

        # Upsample, downsample and output steps
        # self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # the mode nearest (algorithm of interpolation) is deterministic, ensuring reproducibility
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.downsample = nn.MaxPool2d(2, 2)

        # no poolings layers
        self.npl1 = LevelBlock(sequences, widths[0], widths[0])
        self.npl2 = LevelBlock(widths[0], widths[1]//2, widths[1])
        self.npl3 = LevelBlock(widths[1], widths[2]//2, widths[2])
        self.npl4 = LevelBlock(widths[2], widths[3]//2, widths[3])

        # Output
        if self.deep_supervision:
            self.input1 = nn.Sequential(
                conv1x1(widths[0], regions)
            )
            self.out_npl1 = nn.Sequential(
                conv1x1(widths[0], regions)
            )
            self.out_npl2 = nn.Sequential(
                conv1x1(widths[1], regions)
            )
            self.out_npl3 = nn.Sequential(
                conv1x1(widths[2], regions)
            )
            self.out_npl4 = nn.Sequential(
                conv1x1(widths[3], regions)
            )
            self.output3 = nn.Sequential(
                nn.ConvTranspose2d(widths[1], widths[1], kernel_size=4, stride=4),
                conv1x1(widths[1], regions)
            )
            self.output2 = nn.Sequential(
                nn.ConvTranspose2d(widths[0], widths[0], kernel_size=2, stride=2),
                conv1x1(widths[0], regions)
            )
        self.output1 = conv1x1(widths[0] // 2, regions)

        self.weights_initialization()

    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # no pooling layers
        npl1 = self.npl1(x)
        npl2 = self.npl2(npl1)
        npl3 = self.npl3(npl2)
        npl4 = self.npl4(npl3)

        # Encoding phase
        e1 = self.encoder1(x)
        p1 = self.downsample(e1)
        e2 = self.encoder2(p1)
        p2 = self.downsample(e2)
        e3 = self.encoder3(p2)
        p3 = self.downsample(e3)
        e4 = self.encoder4(p3)

        # Bottleneck phase
        bottleneck = self.bottleneck(e4)
        bottleneck2 = self.bottleneck2(torch.cat([e4, bottleneck], dim=1))

        # Decoding phase + skip connections
        up3 = self.upsample(bottleneck2)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample(d2)
        d1 = self.decoder1(torch.cat([e1, up1, npl4], dim=1))

        # Output
        if self.deep_supervision:
            input1 = self.input1(e1)
            out_npl1 = self.out_npl1(npl1)
            out_npl2 = self.out_npl2(npl2)
            out_npl3 = self.out_npl3(npl3)
            out_npl4 = self.out_npl4(npl4)
            output3 = self.output3(d3)
            output2 = self.output2(d2)
            output1 = self.output1(d1)

            return [output3, output2, out_npl1, out_npl2, out_npl3, out_npl4, input1, output1]  # 0.8215269708075569
        else:
            output1 = self.output1(d1)

            return output1


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    seq_ouput = torch.rand(1, 1, 128, 128)

    model = FSB_BTS_UNet(sequences=1, regions=1, width=6, deep_supervision=False)
    preds = model(seq_input)

    print(seq_input.shape)
    if model.deep_supervision:
        for p in preds:
            print(p.shape)
            assert seq_ouput.shape == p.shape
    else:
        print(preds.shape)
        assert seq_ouput.shape == preds.shape

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
