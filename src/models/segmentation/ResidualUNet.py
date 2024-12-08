"""
Code got from: https://github.com/DonDzundza/Pytorch-3D-Medical-Image-Semantic-Segmentation

Zhang, Z., Zhao, T., Gay, H., Zhang, W., & Sun, B. (2020). ARPM‐net:
A novel CNN‐based adversarial method with Markov Random Field enhancement for prostate and
organs at risk segmentation in pelvic CT images. Medical Physics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class in_block(nn.Module):
    """
    in_block is used to connect the input of the whole network.
    number of channels is changed by conv1, and then it keeps the same for all
    following layers.
    parameters:
        channel_in: int
            the number of channels of the input.
            RGB images have 3, greyscale images have 1, etc.
        channel_out: int
            the number of filters for conv1; keeps unchanged for all layers following
            conv1
    """

    def __init__(self, channel_in, channel_out):
        super(in_block, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out

        self.conv1 = nn.Conv2d(
            kernel_size=3,
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=self.channel_out)

        self.conv2 = nn.Conv2d(
            kernel_size=3,
            in_channels=self.channel_out,
            out_channels=self.channel_out,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            kernel_size=3,
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=self.channel_out)

    def forward(self, x):
        path = self.conv1(x)
        path = self.bn1(path)
        path = F.leaky_relu(path)
        path = F.dropout(path, p=0.2)

        path = self.conv2(path)

        residual = self.conv3(x)
        residual = self.bn3(residual)

        self.down_level1 = path + residual

        return self.down_level1


class res_block(nn.Module):
    '''
    res_block used for down and up, toggled by downsample.
    "input" -> bn1 -> relu1 -> conv1 -> bn2 -> relu2 -> conv2 -> "path"
            -> conv3 -> bn3 -> "residual"

    return "output" = "path" + "residual"
    downsampling (if any) is done by conv1
    parameters:
        channel_in: int
        downsample: boolean
            if downsample is true, the block is used for encoding path,
            during which the channels out are doubled by the conv1.
            conv1 will have stride 2.
            if downsample is false, the block is used for segmenting/restoring
            path, during which the channels keep the same through the block.
            conv1 will have stride 1.
    '''

    def __init__(
            self,
            channel_in,
            downsample=False,
    ):
        super(res_block, self).__init__()

        self.channel_in = channel_in

        if downsample:
            self.channel_out = 2 * self.channel_in
            self.conv1_stride = 2
            self.conv3_stride = 2
        else:
            self.channel_out = self.channel_in
            self.conv1_stride = 1
            self.conv3_stride = 1

        self.bn1 = nn.BatchNorm2d(num_features=self.channel_in)
        self.conv1 = nn.Conv2d(
            in_channels=self.channel_in,
            kernel_size=3,
            out_channels=self.channel_out,
            stride=self.conv1_stride,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=self.channel_out)
        self.conv2 = nn.Conv2d(
            in_channels=self.channel_out,
            out_channels=self.channel_out,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=self.channel_in,
            out_channels=self.channel_out,
            stride=self.conv3_stride,
            padding=1,
            kernel_size=3
        )
        self.bn3 = nn.BatchNorm2d(num_features=self.channel_out)

    def forward(self, x):

        path = self.bn1(x)
        path = F.leaky_relu(path)
        path = F.dropout(path, p=0.2)

        path = self.conv1(path)
        path = self.bn2(path)
        path = F.leaky_relu(path)
        path = F.dropout(path, p=0.2)

        path = self.conv2(path)

        residual = self.conv3(x)
        residual = self.bn3(residual)

        output = path + residual

        return output


class encoder(nn.Module):
    """
    encoder
    dataflow:
    x --down_block2--> down_level2
    --down_block3--> down_level3
    --down_block4--> codes
    parameters:
        base_filters: number of filters received from in_block; 16 by default.
    """

    def __init__(self, base_filters):
        super(encoder, self).__init__()

        self.bf = base_filters
        self.name = "ResidualUnet"
        self.down_block2 = res_block(
            channel_in=self.bf,
            downsample=True
        )
        self.down_block3 = res_block(
            channel_in=self.bf * 2,
            downsample=True
        )
        self.down_block4 = res_block(
            channel_in=self.bf * 4,
            downsample=True
        )

    def forward(self, x):
        self.down_level2 = self.down_block2(x)
        self.down_level3 = self.down_block3(self.down_level2)
        self.codes = self.down_block4(self.down_level3)

        return self.codes


class decoder(nn.Module):
    """
    decoder
    dataflow:
    x  --upsample3--> up3 --up_block3--> up_level3
    --upsample2--> up2 --up_block2--> up_level2
    --upsample1--> up1 --up_block1--> up_level1
    parameters:
        base_filters: number of filters consistent with encoder; 16 by default.
    """

    def __init__(
            self,
            base_filters
    ):
        super(decoder, self).__init__()
        self.bf = base_filters

        self.upsample3 = nn.ConvTranspose2d(
            in_channels=self.bf * 8,
            out_channels=self.bf * 4,
            kernel_size=2,
            stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=self.bf * 8,
            out_channels=self.bf * 4,
            kernel_size=1
        )
        self.up_block3 = res_block(
            channel_in=self.bf * 4,
            downsample=False
        )

        self.upsample2 = nn.ConvTranspose2d(
            in_channels=self.bf * 4,
            out_channels=self.bf * 2,
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.bf * 4,
            out_channels=self.bf * 2,
            kernel_size=1
        )
        self.up_block2 = res_block(
            channel_in=self.bf * 2,
            downsample=False
        )

        self.upsample1 = nn.ConvTranspose2d(
            in_channels=self.bf * 2,
            out_channels=self.bf,
            kernel_size=2,
            stride=2
        )
        self.conv1 = nn.Conv2d(
            in_channels=self.bf * 2,
            out_channels=self.bf,
            kernel_size=1
        )
        self.up_block1 = res_block(
            channel_in=self.bf,
            downsample=False
        )

    def forward(self, x):
        up3 = self.upsample3(x)
        self.up_level3 = self.up_block3(up3)

        up2 = self.upsample2(self.up_level3)
        self.up_level2 = self.up_block2(up2)

        up1 = self.upsample1(self.up_level2)
        self.up_level1 = self.up_block1(up1)

        return self.up_level1


class seg_out_block(nn.Module):
    """
    seg_out_block, receive data from decoder and output the segmentation mask
    parameters:
        base_filters: number of filters received from in_block.
        n_classes: number of classes
    """

    def __init__(self, base_filters, n_classes=6):
        super(seg_out_block, self).__init__()

        self.bf = base_filters
        self.n_classes = n_classes
        self.conv = nn.Conv2d(
            in_channels=self.bf,
            out_channels=self.n_classes,
            kernel_size=1
        )

    def forward(self, x):
        self.output = self.conv(x)
        return self.output


class seg_path(nn.Module):
    def __init__(
            self,
            in_block,
            encoder,
            decoder,
            seg_out_block
    ):
        super(seg_path, self).__init__()

        self.in_block = in_block
        self.encoder = encoder
        self.decoder = decoder
        self.seg_out_block = seg_out_block

    def forward(self, x):
        self.down_level1 = self.in_block(x)

        self.down_level2 = self.encoder.down_block2(self.down_level1)
        self.down_level3 = self.encoder.down_block3(self.down_level2)
        self.codes = self.encoder.down_block4(self.down_level3)

        self.up3 = self.decoder.upsample3(self.codes)
        up3_dummy = torch.cat([self.up3, self.down_level3], 1)
        up3_dummy = self.decoder.conv3(up3_dummy)
        self.up_level3 = self.decoder.up_block3(up3_dummy)

        self.up2 = self.decoder.upsample2(self.up_level3)
        up2_dummy = torch.cat([self.up2, self.down_level2], 1)
        up2_dummy = self.decoder.conv2(up2_dummy)
        self.up_level2 = self.decoder.up_block2(up2_dummy)

        self.up1 = self.decoder.upsample1(self.up_level2)
        up1_dummy = torch.cat([self.up1, self.down_level1], 1)
        up1_dummy = self.decoder.conv1(up1_dummy)
        self.up_level1 = self.decoder.up_block1(up1_dummy)

        self.output = self.seg_out_block(self.up_level1)

        return self.output


class ResidualUNet(nn.Module):
    """
    This class implements a variation of 2D Unet network. Main modifications are:
        - Replacement of ReLU activation layer by LeakyReLU
        - Use of instance normalization to ensure a normalization by each sequence

    """

    name = "Residual UNet"

    def __init__(self, sequences=1, regions=1, width=24):
        super(ResidualUNet, self).__init__()

        self.in_block = in_block(channel_in=sequences, channel_out=width)
        self.encoder = encoder(base_filters=width)
        self.decoder = decoder(base_filters=width)
        self.out_block = seg_out_block(base_filters=width, n_classes=regions)

    def forward(self, x):
        x = self.in_block(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out_block(x)

        return x


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    seq_ouput = torch.rand(1, 1, 128, 128)

    model = ResidualUNet(sequences=1, regions=1, width=24)
    preds = model(seq_input)

