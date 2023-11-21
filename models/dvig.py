"""
Implementation of of D-VIG as described in our midterm report.
"""
import math
import sys
import torch
import torch.nn as nn
import torchaudio.transforms as audiotransforms

import models.modules as modules
from models.modules_diffusion import Diffusion


class DiffusionVIG(nn.Module):
    def __init__(
        self,
        is_grayscale: bool = True,
        use_resnet: bool = True,
        hidden_size: int = 20,
        num_lstm_layers: int = 2,
        num_diffusion_timesteps: int = 5,
        n_fft: int = 400,
        noise_steps=20,
        device="cpu",
    ):
        super(DiffusionVIG, self).__init__()
        if n_fft % 2 == 1:
            n_fft += 1
        lstm_hidden_size = int(n_fft / 2) + 1
        lstm_output_size = 2 * lstm_hidden_size  # x2 since we fold it as complex tensor
        diffusion_size = 2 * int(n_fft / 2)

        self.cnn = modules.VideoCNN(
            hidden_size, use_resnet=use_resnet, is_grayscale=is_grayscale
        )
        self.lstm = modules.VideoLSTM(
            lstm_hidden_size, lstm_output_size, num_lstm_layers
        )

        self.diffusion = Diffusion(
            img_size=diffusion_size, noise_steps=noise_steps, device=device
        )

        self.istft = audiotransforms.InverseSpectrogram(n_fft=n_fft)
        self.num_timesteps = num_diffusion_timesteps

    def forward(self, x, _):
        """
        :param x: (batch_size, seq_len, height, width) video frames
        :param _: unused audio waveforms
        """
        t = self.diffusion.sample_timesteps(x.shape[0])
        x, noise = self.diffusion.noise_images(x, t)  # TODO: MAKE NOISE_VIDEOS

        # Fetch final hidden state after running video through cnn+lstm
        x = self.cnn(x)
        x = self.lstm(x)

        # "fold" x into (batch_size, seq_len, hidden_size) so we get real and imaginary parts
        x_real, x_imag = x.chunk(2, dim=-1)
        x = torch.complex(x_real, x_imag)

        x = self.synthesize_audiowave(x)

        return x

    def synthesize_audiowave(self, spectrogram: torch.Tensor):
        # spectrogram shape: (batch_size, seq_len, dim)
        print(spectrogram.shape)
        spectrogram = spectrogram.permute(0, 2, 1)
        return self.istft(spectrogram)


if __name__ == "__main__":
    batch_size, seq_len, height, width = 2, 10, 24, 36
    hidden_size = 20
    model = DiffusionVIG(hidden_size=hidden_size)
    x = torch.rand([batch_size, seq_len, height, width])
    y = model(x, None)
    print(y.shape)
