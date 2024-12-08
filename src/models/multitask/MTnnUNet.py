import torch
from torch import nn
from collections import OrderedDict


def conv1x1(in_channels, out_channels):
    """ 3D convolution which uses a kernel size of 1"""

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=False):
    """ 3D convolution which uses a kernel size of 3"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


class ConvInNormLeReLU(nn.Sequential):
    """
    This class stacks a 3D Convolution, Instance Normalization and Leaky ReLU layers.

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
    This class stacks two blocks of ConvInNormLeReLU (3D Convolution, Instance Normalization and Leaky ReLU layers).

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


class MTnnUNet(nn.Module):
    """
    This class extends nnU-Net model for Multi-task segmentation and classification
    """

    def __init__(self, sequences, regions, n_classes=3):
        super().__init__()

        widths = [32, 64, 128, 256, 320]

        self.n_classes = n_classes
        if n_classes == 2:
            self.n_classes = 1

        # Encoders
        self.encoder1 = LevelBlock(sequences, widths[0], widths[0])
        self.encoder2 = LevelBlock(widths[0], widths[1], widths[1])
        self.encoder3 = LevelBlock(widths[1], widths[2], widths[2])
        self.encoder4 = LevelBlock(widths[2], widths[3], widths[3])
        self.encoder5 = LevelBlock(widths[3], widths[4], widths[4])

        # Bottleneck
        self.bottleneck = LevelBlock(widths[4], widths[4], widths[4])

        # Decoders
        self.decoder5 = LevelBlock(widths[4] + widths[4], widths[3], widths[3])
        self.decoder4 = LevelBlock(widths[3] + widths[3], widths[2], widths[2])
        self.decoder3 = LevelBlock(widths[2] + widths[2], widths[1], widths[1])
        self.decoder2 = LevelBlock(widths[1] + widths[1], widths[0], widths[0])
        self.decoder1 = LevelBlock(widths[0] + widths[0], widths[0], widths[0] // 2)

        # Upsample layers
        self.upsample5 = nn.ConvTranspose2d(widths[4], widths[4], kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(widths[3], widths[3], kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(widths[2], widths[2], kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(widths[1], widths[1], kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(widths[0], widths[0], kernel_size=2, stride=2)

        # Downsample and output steps
        self.downsample = nn.MaxPool2d(2, 2)

        # Outputs
        self.output4 = nn.Sequential(
            nn.ConvTranspose2d(widths[2], widths[2], kernel_size=8, stride=8),
            conv1x1(widths[2], regions)
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

        # Define classification decoder layers
        self.process_encoder_5 = ConvInNormLeReLU(widths[4], widths[4])
        self.process_decoder_5 = ConvInNormLeReLU(widths[3], widths[4])
        self.classifier = nn.Sequential(
            ConvInNormLeReLU(widths[4] * 3, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # Flatten output
            nn.Linear(512, 256),  # Fully-connected layer
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.n_classes)  # Classification output
        )

    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        e5 = self.encoder5(p4)
        p5 = self.downsample(e5)

        # Bottleneck
        bottleneck = self.bottleneck(p5)

        # Decoding phase + skip connections
        up5 = self.upsample5(bottleneck)
        d5 = self.decoder5(torch.cat([e5, up5], dim=1))
        up4 = self.upsample4(d5)
        d4 = self.decoder4(torch.cat([e4, up4], dim=1))
        up3 = self.upsample3(d4)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample2(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample1(d2)
        d1 = self.decoder1(torch.cat([e1, up1], dim=1))

        """
        Classifier
        """
        features_extracted = torch.cat([self.process_encoder_5(e5), self.upsample5(bottleneck), self.process_decoder_5(d5)], dim=1)
        predicted_class = self.classifier(features_extracted)

        # Output
        output4 = self.output4(d4)
        output3 = self.output3(d3)
        output2 = self.output2(d2)
        output1 = self.output1(d1)

        return [predicted_class], [output4, output3, output2, output1]


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    seq_output = torch.rand(1, 1, 128, 128)

    model = MTnnUNet(sequences=1, regions=1, n_classes=3)
    logits, segmentation = model(seq_input)
