import torch
from torch import nn
import torch.nn.functional as F
import math
from scipy.ndimage import label
import numpy as np
from .utils import normalize_attention, normalize_and_smooth_attention



# class FusionNet(nn.Module):
#     def __init__(self, in_ch=3, out_ch=1):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(in_ch, 8, 3, 1, 1),
#             nn.InstanceNorm3d(8),
#             nn.LeakyReLU(),
#             nn.Conv3d(8, 16, 3, 1, 1),
#             nn.InstanceNorm3d(16),
#             nn.LeakyReLU())
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(in_ch + 16, 16, 3, 1, 1),
#             nn.LeakyReLU(),
#             nn.Conv3d(16, out_ch, 3, 1, 1),
#             nn.ReLU())

#     def forward(self, x):
#         # return self.conv2(x + self.conv1(x))
#         return self.conv2(torch.cat([x, self.conv1(x)], dim=1))

class UNet3d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        conditional_ch=0,
        num_lvs=4,
        base_ch=16,
        final_act="noact"
    ):
        super().__init__()

        self.final_act = final_act
        self.conditional_ch = conditional_ch

        self.in_conv = nn.Conv3d(
            in_ch,
            base_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for lv in range(num_lvs):

            ch = base_ch * (2 ** lv)

            self.down_convs.append(
                ConvBlock3d(
                    ch + conditional_ch,
                    ch * 2,
                    ch * 2
                )
            )

            self.down_samples.append(
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=2
                )
            )

            self.up_samples.append(
                Upsample3d(ch * 4)
            )

            self.up_convs.append(
                ConvBlock3d(
                    ch * 4,
                    ch * 2,
                    ch * 2
                )
            )

        bottleneck_ch = base_ch * (2 ** num_lvs)

        self.bottleneck_conv = ConvBlock3d(
            bottleneck_ch,
            bottleneck_ch * 2,
            bottleneck_ch * 2
        )

        self.out_conv = nn.Sequential(
            nn.Conv3d(
                base_ch * 2,
                base_ch,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.LeakyReLU(0.1),

            nn.Conv3d(
                base_ch,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, in_tensor, condition=None):

        encoded_features = []

        x = self.in_conv(in_tensor)

        for down_conv, down_sample in zip(
            self.down_convs,
            self.down_samples
        ):

            if condition is not None:

                # x:
                # [B, C, D, H, W]

                condition_expanded = condition.expand(
                    -1,
                    -1,
                    x.shape[2],
                    x.shape[3],
                    x.shape[4]
                )

                x = torch.cat(
                    [x, condition_expanded],
                    dim=1
                )

            down_conv_out = down_conv(x)

            x = down_sample(down_conv_out)

            encoded_features.append(down_conv_out)

        x = self.bottleneck_conv(x)

        for encoded_feature, up_conv, up_sample in zip(
            reversed(encoded_features),
            reversed(self.up_convs),
            reversed(self.up_samples)
        ):

            x = up_sample(
                x,
                encoded_feature
            )

            x = up_conv(x)

        x = self.out_conv(x)

        if self.final_act == "sigmoid":
            x = torch.sigmoid(x)

        elif self.final_act == "relu":
            x = torch.relu(x)

        elif self.final_act == "tanh":
            x = torch.tanh(x)

        return x

class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(mid_ch),
            nn.LeakyReLU(0.1),

            nn.Conv3d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


class Upsample3d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        out_ch = in_ch // 2

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, encoded_feature):
        x = F.interpolate(
            x,
            size=encoded_feature.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        x = self.conv(x)

        return torch.cat([encoded_feature, x], dim=1)


class EtaEncoder3d(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                in_ch,
                16,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.1),

            # 96 x 112 x 96

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.1),

            # 48 x 56 x 48

            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.1),

            # 24 x 28 x 24

            nn.Conv3d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.1),

            # 12 x 14 x 12

            nn.Conv3d(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.1),

            # 6 x 7 x 6
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.out = nn.Linear(
            128,
            out_ch
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return self.out(x)
        
class Patchifier3d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=1,
        patch_size=32
    ):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                in_ch,
                64,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0
            ),

            nn.LeakyReLU(0.1),

            nn.Conv3d(
                64,
                out_ch,
                kernel_size=1,
                stride=1
            )
        )

    def forward(self, x):
        return self.conv(x)

# 2D theta encoder uses 17×17×17 convolution. That becomes computationally expensive very quickly.
# For 3D, using repeated stride-2 3x3x3 convolutions and then global average pooling.
class ThetaEncoder3d(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()

        self.conv = nn.Sequential(

            # [B, 1, 192, 224, 192]
            nn.Conv3d(
                in_ch,
                16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.1),

            # [B, 16, 96, 112, 96]
            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.1),

            # [B, 32, 48, 56, 48]
            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.1),

            # [B, 64, 24, 28, 24]
            nn.Conv3d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.1),

            # [B, 128, 12, 14, 12]
            nn.Conv3d(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.1),

            # [B, 128, 6, 7, 6]
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.mean_fc = nn.Linear(
            128,
            out_ch
        )

        self.logvar_fc = nn.Linear(
            128,
            out_ch
        )

    def forward(self, x):

        x = self.conv(x)

        # [B, 128, 1, 1, 1]
        x = self.global_pool(x)

        # [B, 128]
        x = torch.flatten(x, 1)

        mu = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        return mu, logvar

class AttentionModule3d(nn.Module):
    def __init__(self, dim, v_ch=5):
        super().__init__()

        self.dim = dim
        self.v_ch = v_ch

        embed_dim = 16

        self.q_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.k_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.scale = embed_dim ** (-0.5)

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        modality_dropout=None,
        temperature=10.0
    ):
        """
        q:
            [B, dim]

        k:
            [B, N, dim]

        v:
            [B, N, v_ch, D, H, W]

        mask:
            optional [B, N, D, H, W]

        modality_dropout:
            optional [B, N]
        """

        B, N, C, D, H, W = v.shape

        q = self.q_fc(q)
        # [B, 16]

        k = self.k_fc(k)
        # [B, N, 16]

        scores = torch.einsum(
            "bd,bnd->bn",
            q,
            k
        )

        scores = scores * self.scale

        if modality_dropout is not None:
            scores = scores.masked_fill(
                modality_dropout.bool(),
                -1e9
            )

        attention = (
            scores / temperature
        ).softmax(dim=1)

        # Convert global attention to volumetric map
        attention = attention[
            :,
            :,
            None,
            None,
            None
        ].expand(
            -1,
            -1,
            D,
            H,
            W
        )

        # Optional foreground/background weighting
        if mask is not None:

            attention = attention * mask

            attention = attention / (
                attention.sum(
                    dim=1,
                    keepdim=True
                ) + 1e-8
            )

        # Apply attention to beta logits
        fused_v = (
            v * attention[:, :, None]
        ).sum(dim=1)

        # fused_v:
        # [B, v_ch, D, H, W]

        return fused_v, attention  
