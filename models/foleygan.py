"""
Implementation of of "FoleyGAN: Visually Guided Generative Adversarial 
Network-Based Synchronous Sound Generation in Silent Videos".

https://arxiv.org/pdf/2107.09262.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as audiotransforms
import models.modules as modules
from models.modules_biggan import BigGAN
from models.modules_trn import RelationModuleMultiScale


# What is img_feature_dim? Height or width of images. Must receive square images (e.g. 64x64)
class FoleyGAN(nn.Module):
    def __init__(
        self,
        img_feature_dim,
        num_class,
        hidden_size,
        batch_size,
        biggan_z_dim: int = 128,
        n_fft: int = 400,
        is_grayscale: bool = True,
    ):
        super(FoleyGAN, self).__init__()
        GAN_OUTPUT_DIM = 128
        MULTI_SCALE_NUM_FRAMES = 8
        NUM_FRAMES = 3
        self.sequence_length = int(6030/batch_size)
        self.n_fft = n_fft
        self.biggan_z_dim = biggan_z_dim

        self.cnn = modules.VideoCNN(
            img_feature_dim, use_resnet=True, is_grayscale=is_grayscale
        )

        trn_output_dim = n_fft // 2 + 1
        self.trn = RelationModuleMultiScale(
            img_feature_dim, num_frames=NUM_FRAMES, num_class=trn_output_dim
        )
        self.mtrn = RelationModuleMultiScale(
            img_feature_dim, num_frames=MULTI_SCALE_NUM_FRAMES, num_class=num_class
        )

        self.biggan = BigGAN(
            hidden_size,
            NUM_FRAMES,
            n_fft,
            z_dim=biggan_z_dim,
            gan_output_dim=GAN_OUTPUT_DIM,
        )
        self.istft = audiotransforms.InverseSpectrogram(n_fft)
        self.stft = audiotransforms.Spectrogram(n_fft)
        self.discriminator = modules.Discriminator(self.sequence_length, 50)

        # Outputs to be saved for loss calculations
        self.toggle_freeze_discriminator()
        self.discrim_loss_fn = nn.HingeEmbeddingLoss()
        self.x_pred = None
        self.x_discrim = None

    def forward(self, x, _):
        batch_size = x.shape[0]

        x = self.cnn(x)

        # Generate audio waveform with biggan
        x_class = self.mtrn(x)
        x_spectrogram = self.trn(x)
        noise = torch.rand(batch_size, self.biggan_z_dim)
        x = self.biggan(noise, x_class, x_spectrogram)

        # Create audio wave via istft
        x = x.permute(0, 2, 1)
        z = x.reshape([x.shape[0], -1])
        z = torch.cat((z.real, z.imag))
        z = z[:, :, None]
        z = z.permute(2, 1, 0)
        self.x_discrim = self.discriminator(z)
        x = self.istft(x)
        x = x.reshape([*x.shape, 1])  # (bs, seq_len, 1)

        self.x_pred = x
        return x

    def toggle_freeze_generator(self):
        for param in self.cnn.parameters():
            param.requires_grad = not(param.requires_grad)

        for param in self.mtrn.parameters():
            param.requires_grad = not(param.requires_grad)

        for param in self.trn.parameters():
            param.requires_grad = not(param.requires_grad)

        for param in self.biggan.parameters():
            param.requires_grad = not(param.requires_grad)

    def toggle_freeze_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = not(param.requires_grad)

    def loss(self, outputs, _, audiowaves):
        """Function starts with discriminator frozen, generator unfrozen"""
        batch_size, pred_seq_len, dimension = outputs.shape

        # Train discriminator on negative and positive example
        self.toggle_freeze_discriminator()
        self.toggle_freeze_generator()

        # Negative example
        loss_discriminator = self.discrim_loss_fn(
            self.x_discrim, torch.zeros((batch_size, 1))
        )
        loss_discriminator.backward()

        # Positive example
        target_size = self.sequence_length
        audiowaves_downsampled = F.interpolate(audiowaves.unsqueeze(1), size=target_size, mode='linear', align_corners=False)
        audiowaves_downsampled = audiowaves_downsampled.squeeze(1)
        print(audiowaves_downsampled.shape)
        spectrogram = self.stft(audiowaves_downsampled)
        print(spectrogram.shape)
        spectrogram = spectrogram.reshape([spectrogram.shape[0], -1])
        spectrogram = spectrogram[:, :, None]
        spectrogram = spectrogram.permute(2, 1, 0)
        print(spectrogram.shape)
        x_discrim_raw = self.discriminator(spectrogram)
        loss_discriminator = self.discrim_loss_fn(
            x_discrim_raw, torch.ones((batch_size, 1))
        )
        loss_discriminator.backward()

        self.toggle_freeze_discriminator()
        self.toggle_freeze_generator()
        loss_generator = modules.calculate_audiowave_loss(audiowaves, outputs)
        return loss_generator
