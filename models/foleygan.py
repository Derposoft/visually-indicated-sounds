"""
Implementation of of "FoleyGAN: Visually Guided Generative Adversarial 
Network-Based Synchronous Sound Generation in Silent Videos".

https://arxiv.org/pdf/2107.09262.pdf
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio.transforms as audiotransforms
import models.modules as modules
from models.TRNmodule import RelationModule, RelationModuleMultiScale
from pytorch_pretrained_biggan import (one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

# What is img_feature_dim? Height or width of images. Must receive square images (e.g. 64x64)
class foleygan(nn.Module):
    def __init__(
        self, 
        img_feature_dim,
        num_class,
        hidden_size,
        n_fft,
        is_grayscale: bool = True
        ):
        super(foleygan, self).__init__()
        self.truncation = 0.4
        TWO_FRAME_TRN = 2
        MULTI_SCALE_NUM_FRAMES = 8
        MAX_NUM_FRAMES = 3

        self.cnn = modules.VideoCNN(img_feature_dim, use_resnet=True, is_grayscale=is_grayscale)
        
        self.trn = RelationModuleMultiScale(img_feature_dim, num_frames=TWO_FRAME_TRN, num_class=num_class)
        self.mtrn = RelationModuleMultiScale(img_feature_dim, num_frames=MULTI_SCALE_NUM_FRAMES, num_class=num_class)

        self.fc1 = nn.Linear(num_class, num_class) # Output of mtrn is size num_class

        self.spectrogram = audiotransforms.Spectrogram(n_fft)

        self.biggan = modules.BigGAN(hidden_size, MAX_NUM_FRAMES, n_fft)

        self.istft = audiotransforms.InverseSpectrogram(n_fft)

        self.discriminator = modules.Discriminator(1, 50)

    def forward(self, x, _):
        x_resnet50 = self.cnn(x)

        x_mtrn = self.mtrn(x_resnet50)
        x_trn = self.trn(x_resnet50)

        x_spectrogram = self.spectrogram(x_trn)
        x_class = self.fc1(x_mtrn)

        noise_vector = truncated_noise_sample(truncation=self.truncation, batch_size=1)
        noise_vector = torch.from_numpy(noise_vector)

        x_biggan = self.biggan(noise_vector, x_class, self.truncation, x_spectrogram)

        x = self.istft(x_biggan)
        
        #x_discriminator = self.discriminator(x)

        return x