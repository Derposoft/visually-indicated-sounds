import logging
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 15

# coding: utf-8
""" BigGAN PyTorch model.
    From "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
    By Andrew Brocky, Jeff Donahuey and Karen Simonyan.
    https://openreview.net/forum?id=B1xsqj09Fm

    PyTorch version implemented from the computational graph of the TF Hub module for BigGAN.
    Some part of the code are adapted from https://github.com/brain-research/self-attention-gan

    This version only comprises the generator (since the discriminator's weights are not released).
    This version only comprises the "deep" version of BigGAN (see publication).
"""

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "biggan-deep-128": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin",
    "biggan-deep-256": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin",
    "biggan-deep-512": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin",
}

PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "biggan-deep-128": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json",
    "biggan-deep-256": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json",
    "biggan-deep-512": "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-config.json",
}

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


def sn_embedding(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(**kwargs), eps=eps)


class SelfAttn(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            bias=False,
            eps=eps,
        )
        self.snconv1x1_phi = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            bias=False,
            eps=eps,
        )
        self.snconv1x1_g = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            bias=False,
            eps=eps,
        )
        self.snconv1x1_o_conv = snconv2d(
            in_channels=in_channels // 2,
            out_channels=in_channels,
            kernel_size=1,
            bias=False,
            eps=eps,
        )
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma * attn_g
        return out


class GenBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        reduction_factor=4,
        up_sample=False,
        eps=1e-12,
    ):
        super(GenBlock, self).__init__()
        self.up_sample = up_sample
        self.drop_channels = in_size != out_size
        middle_size = in_size // reduction_factor

        self.bn_0 = nn.BatchNorm2d(in_size)
        self.conv_0 = snconv2d(
            in_channels=in_size, out_channels=middle_size, kernel_size=1, eps=eps
        )

        self.bn_1 = nn.BatchNorm2d(middle_size)
        self.conv_1 = snconv2d(
            in_channels=middle_size,
            out_channels=middle_size,
            kernel_size=3,
            padding=1,
            eps=eps,
        )

        self.bn_2 = nn.BatchNorm2d(middle_size)
        self.conv_2 = snconv2d(
            in_channels=middle_size,
            out_channels=middle_size,
            kernel_size=3,
            padding=1,
            eps=eps,
        )

        self.bn_3 = nn.BatchNorm2d(middle_size)
        self.conv_3 = snconv2d(
            in_channels=middle_size, out_channels=out_size, kernel_size=1, eps=eps
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = x

        x = self.bn_0(x)
        x = self.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x)
        x = self.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_1(x)

        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv_2(x)

        x = self.bn_3(x)
        x = self.relu(x)
        x = self.conv_3(x)

        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode="nearest")

        out = x + x0
        return out


class Generator(nn.Module):
    def __init__(self, config, hidden_size):
        super(Generator, self).__init__()
        self.config = config
        ch = config.channel_width
        condition_vector_dim = config.z_dim * 2 + hidden_size

        self.gen_z = snlinear(
            in_features=condition_vector_dim,
            out_features=4 * 4 * 16 * ch,
            eps=config.eps,
        )

        layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                layers.append(SelfAttn(ch * layer[1], eps=config.eps))
            layers.append(
                GenBlock(
                    ch * layer[1],
                    ch * layer[2],
                    up_sample=layer[0],
                    eps=config.eps,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU()
        self.conv_to_rgb = snconv2d(
            in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=config.eps
        )
        self.tanh = nn.Tanh()

    def forward(self, cond_vector):
        z = self.gen_z(cond_vector)

        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            z = layer(z)

        z = self.bn(z)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, ...]
        z = self.tanh(z)

        return z


class BigGAN(nn.Module):
    """BigGAN Generator."""

    def __init__(
        self,
        hidden_size,
        n_frames,
        n_fft,
        z_dim=128,
        gan_output_dim=128,
        audio_sample_rate_out=90,
    ):
        super(BigGAN, self).__init__()
        self.gan_output_dim = gan_output_dim
        self.n_frames = n_frames
        self.audio_sample_rate_out = audio_sample_rate_out
        self.spectrogram_len = audio_sample_rate_out // n_frames
        config = BigGANConfig(output_dim=gan_output_dim, z_dim=z_dim)
        spectrogram_dim = n_fft // 2 + 1  # n_frames * (n_fft // 2 + 1)
        self.config = config
        self.embeddings = nn.Linear(config.num_classes, config.z_dim, bias=False)
        self.generator = Generator(config, hidden_size)
        self.linear = nn.Linear(spectrogram_dim, hidden_size)
        self.linear2 = nn.Linear(
            3 * gan_output_dim**2, 2 * spectrogram_dim * self.spectrogram_len
        )

    def forward(self, z, class_label, spectrogram):
        # Our preprocessing logic for biggan=
        spectrogram = spectrogram.reshape([spectrogram.shape[0], -1])
        embed = self.embeddings(class_label)
        spectrogram = self.linear(spectrogram)
        cond_vector = torch.cat((z, embed, spectrogram), dim=1)

        # biggan, outputs (batch_size, 3, output_dim, output_dim)
        z = self.generator(cond_vector)

        # Our postprocessing logic for biggan
        z = z.reshape([z.shape[0], -1])
        z = self.linear2(z)
        z_real, z_imag = z.chunk(2, dim=-1)
        z = torch.complex(z_real, z_imag)
        z = z.reshape([z.shape[0], self.spectrogram_len, -1])  # (bs, seq_len, dim)
        return z


# coding: utf-8
"""
BigGAN config.
"""


class BigGANConfig(object):
    """Configuration class to store the configuration of a `BigGAN`.
    Defaults are for the 128x128 model.
    layers tuple are (up-sample in the layer ?, input channels, output channels)
    """

    def __init__(
        self,
        output_dim=128,
        z_dim=128,
        class_embed_dim=128,
        channel_width=128,
        num_classes=NUM_CLASSES,
        layers=[
            (False, 16, 16),
            (True, 16, 16),
            (False, 16, 16),
            (True, 16, 8),
            (False, 8, 8),
            (True, 8, 4),
            (False, 4, 4),
            (True, 4, 2),
            (False, 2, 2),
            (True, 2, 1),
        ],
        attention_layer_position=8,
        eps=1e-4,
        n_stats=51,
    ):
        """Constructs BigGANConfig."""
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.channel_width = channel_width
        self.num_classes = num_classes
        self.layers = layers
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BigGANConfig` from a Python dictionary of parameters."""
        config = BigGANConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BigGANConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
