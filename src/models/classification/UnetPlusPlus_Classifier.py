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

from typing import Sequence, Union
import logging
import torch
import torch.nn as nn

from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep


class UNetPlusPlusClassifier(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        n_classes: int = 3,
        features: Sequence[int] = (24, 48, 96, 192, 384, 24),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
    ):
        """
        A UNet++ implementation with 1D/2D/3D supports.

        Based on:

            Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image
            Segmentation". 4th Deep Learning in Medical Image Analysis (DLMIA)
            Workshop, DOI: https://doi.org/10.48550/arXiv.1807.10165


        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
        #     >>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))
        #
        #     # for spatial 2D, with deep supervision enabled
        #     >>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), deep_supervision=True)
        #
        #     # for spatial 2D, with group norm
        #     >>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))
        #
        #     # for spatial 3D
        #     >>> net = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
        #
        # See Also
            - :py:class:`monai.networks.nets.BasicUNet`
            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        self.n_classes = n_classes
        if n_classes == 2:
            self.n_classes = 1

        fea = ensure_tuple_rep(features, 6)\

        logging.info(f"BasicUNetPlusPlus features: {fea}.")

        # Features extractor
        self.conv_0_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.conv_1_0 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.conv_2_0 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.conv_3_0 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.conv_4_0 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.upcat_3_1 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)

        # Classifier
        self.softmax = nn.Softmax(dim=1)
        self.process_level_3 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.classifier = nn.Sequential(
            TwoConv(spatial_dims, fea[4] * 3, 512, act, norm, bias, dropout),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # Flatten output
            nn.Linear(512, 256),  # Fully-connected layer
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.n_classes)  # Classification output
        )

        # self.softmax = nn.Softmax(dim=1)
        # self.process_level_3 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        # self.classifier = nn.Sequential(
        #     TwoConv(spatial_dims, fea[4] * 3, fea[1], act, norm, bias, dropout),
        #     nn.AvgPool2d(kernel_size=8),
        #     nn.Flatten(),  # Flatten output
        #     nn.Linear(fea[1], 16),  # Fully-connected layer
        #     nn.ReLU(),
        #     nn.Dropout(.2),
        #     nn.Linear(in_features=16, out_features=self.n_classes)  # Classification output
        # )


    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """

        # Encoder
        x_0_0 = self.conv_0_0(x)
        x_1_0 = self.conv_1_0(x_0_0)
        x_2_0 = self.conv_2_0(x_1_0)
        x_3_0 = self.conv_3_0(x_2_0)
        x_4_0 = self.conv_4_0(x_3_0)
        x_3_1 = self.upcat_3_1(x_4_0, x_3_0)

        # Classifier
        features_extracted = torch.cat([self.process_level_3(x_3_0), x_4_0, self.process_level_3(x_3_1)], dim=1)
        predicted_class = self.classifier(features_extracted)
        # if self.n_classes > 2:
        #     predicted_class = self.softmax(predicted_class)

        output = predicted_class

        return output


if __name__ == "__main__":
    seq_input = torch.rand(1, 1, 128, 128)
    seq_ouput = torch.rand(1, 1, 128, 128)

    model = UNetPlusPlusClassifier(spatial_dims=2, in_channels=1, n_classes=3)
    pred = model(seq_input)
    print(pred.shape)
    print(seq_input.shape)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))