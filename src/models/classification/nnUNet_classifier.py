# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


class nnUNetClassifier(nn.Module):
    """
    This class implements a variation of 3D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "nn-UNet2021"

    def __init__(self, sequences, n_classes=3):
        super(nnUNetClassifier, self).__init__()

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

        # Upsamplers
        self.upsample5 = nn.ConvTranspose2d(widths[4], widths[4], kernel_size=2, stride=2)

        # Downsample and output steps
        self.downsample = nn.MaxPool2d(2, 2)

        self.weights_initialization()

        # Define classification decoder layers
        self.softmax = nn.Softmax(dim=1)
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


        """
        Classifier
        """
        features_extracted = torch.cat([self.process_encoder_5(e5), self.upsample5(bottleneck), self.process_decoder_5(d5)], dim=1)
        predicted_class = self.classifier(features_extracted)
        if self.n_classes > 2:
            predicted_class = self.softmax(predicted_class)

        return predicted_class


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    seq_ouput = torch.rand(1, 1, 128, 128)

    model = nnUNetClassifier(sequences=1, n_classes=3)
    predicted_class = model(seq_input)

