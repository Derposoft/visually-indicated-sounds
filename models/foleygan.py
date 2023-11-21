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
        audio_sample_rate_out: int = 90,
        is_grayscale: bool = True,
    ):
        super(FoleyGAN, self).__init__()
        self.MULTI_SCALE_NUM_FRAMES = 8
        GAN_OUTPUT_DIM = 128
        NUM_FRAMES = 3
        self.sequence_length = int(
            (((n_fft // 2) + 1) * (audio_sample_rate_out / NUM_FRAMES)) / batch_size
        )
        self.stft_downsample = int(
            (((n_fft // 2) + 1) * ((audio_sample_rate_out / NUM_FRAMES) - 1))
            / batch_size
        )
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
            img_feature_dim, num_frames=self.MULTI_SCALE_NUM_FRAMES, num_class=num_class
        )

        self.biggan = BigGAN(
            hidden_size,
            NUM_FRAMES,
            n_fft,
            z_dim=biggan_z_dim,
            gan_output_dim=GAN_OUTPUT_DIM,
            audio_sample_rate_out=audio_sample_rate_out,
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
        n_frames = x.shape[1]

        # Run through CNN and then pad end of sequence if it is too small for MTRN
        x = self.cnn(x)
        if n_frames < self.MULTI_SCALE_NUM_FRAMES:
            x = F.pad(x, (0, 0, 0, self.MULTI_SCALE_NUM_FRAMES - n_frames))

        # Generate audio waveform with biggan
        x_class = self.mtrn(x)
        x_spectrogram = self.trn(x)
        noise = torch.rand(batch_size, self.biggan_z_dim)
        x = self.biggan(noise, x_class, x_spectrogram)

        # Get discriminator output
        x_real, x_imag = x.real, x.imag
        x_discrim = torch.cat([x_real, x_imag], dim=0)
        x_discrim = x_discrim.reshape(x_discrim.shape[0], -1)
        x_discrim = x_discrim.unsqueeze(-1)
        self.x_discrim = self.discriminator(x_discrim)

        # Create audio wave via istft
        x = x.permute(0, 2, 1)
        x = self.istft(x)
        self.x_pred = x
        return x

    def toggle_freeze_generator(self):
        for param in self.cnn.parameters():
            param.requires_grad = not (param.requires_grad)
        for param in self.mtrn.parameters():
            param.requires_grad = not (param.requires_grad)
        for param in self.trn.parameters():
            param.requires_grad = not (param.requires_grad)
        for param in self.biggan.parameters():
            param.requires_grad = not (param.requires_grad)

    def toggle_freeze_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = not (param.requires_grad)

    def loss(self, outputs, _, audiowaves):
        """Function starts with discriminator frozen, generator unfrozen"""
        batch_size, pred_seq_len = outputs.shape
        audiowaves = F.interpolate(
            audiowaves.unsqueeze(1),
            size=pred_seq_len,
            mode="linear",
            align_corners=False,
        )[0]

        # Discriminator loss
        self.toggle_freeze_discriminator()
        self.toggle_freeze_generator()
        spectrogram = self.stft(audiowaves)
        spectrogram = spectrogram.reshape([spectrogram.shape[0], -1])
        spectrogram = spectrogram[:, :, None].permute(2, 1, 0)
        x_discrim_pos = self.discriminator(spectrogram)
        loss_discrim_pos = self.discrim_loss_fn(
            x_discrim_pos, torch.ones((batch_size, 1))
        )
        loss_discrim_neg = self.discrim_loss_fn(
            self.x_discrim, torch.zeros((batch_size, 1))
        )
        loss_discrim = loss_discrim_pos + loss_discrim_neg
        loss_discrim.backward(retain_graph=True)

        # Generator output
        self.toggle_freeze_discriminator()
        self.toggle_freeze_generator()
        loss_generator = 1 - loss_discrim
        return loss_generator


if __name__ == "__main__":
    print("Running foleygan tests....")
    num_classes = 15
    batch_size = 1
    img_feature_dim = 5
    hidden_size = 5
    n_fft = 400
    model = FoleyGAN(img_feature_dim, num_classes, hidden_size, batch_size, n_fft)
    print("Model initialized")

    n_frames = 5
    width = 300
    height = 240
    x = torch.rand(batch_size, n_frames, width, height)
    y = model(x, None)
    print(y.shape)
