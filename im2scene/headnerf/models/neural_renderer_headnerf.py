import torch.nn as nn
import torch
from torchvision import transforms as T
from math import log2
from im2scene.layers import Blur
import numpy as np

class NeuralRenderer(nn.Module):
    def __init__(
            self, n_feat=128, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=32, img_size=64, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False,
            **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_blocks = int(log2(img_size) - 4)

        # Define upsampler for rgb stream
        self.upsample_rgb = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)

        # Register toRGB layers
        self.toRGBs = nn.ModuleList(
            [nn.Conv2d(input_dim, out_dim, (1, 1))] +
            [nn.Conv2d(n_feat // (2 ** (i+1)),
                       out_dim, (1, 1)) for i in range(0, self.n_blocks - 1)]
        )

        # Register feature stream upsamplers
        self.conv_feats = nn.ModuleList(
            [upsampler_feat(self.input_dim // (2**i)) for i in range(0, self.n_blocks - 1)]
        )

    def forward(self, I_F, it=0):

        # Perform first toRGB
        rgb = self.upsample_rgb(self.toRGBs[0](I_F))

        for idx, layer in enumerate(self.conv_feats):

            I_F = layer(I_F)
            rgb = rgb + self.toRGBs[idx+1](I_F)

            if (idx < len(self.conv_feats) - 1):
                rgb = self.upsample_rgb(rgb)

        # TODO: think this is not correct, should not be necessary to do one more upsampling
        rgb = self.upsample_rgb(rgb)

        return rgb

class upsampler_feat(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(dim_in, 2*dim_in, (1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(2*dim_in, 4*dim_in, (1, 1)),
            nn.LeakyReLU()
        )

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.fixed_blur_conv = T.GaussianBlur(kernel_size=3, sigma=0.1)

        # TODO: this behaviour is new to headnerf, not in stylenerf or giraffe?
        self.final_downsizing = nn.Sequential(
            nn.Conv2d(dim_in, dim_in // 2, (1, 1)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x4 = torch.repeat_interleave(x, 4, dim=1)
        x_conv = self.conv_layers(x)

        pixel_shuffled = self.pixel_shuffle(x4 + x_conv)

        out = self.fixed_blur_conv(pixel_shuffled)
        out = self.final_downsizing(out)

        return out
