"""
Implementation of of D-VIG as described in our midterm report.
"""
import math
import sys
import torch
import torch.nn as nn
import torchaudio.transforms as audiotransforms

import models.modules as modules


class DiffusionVIG(nn.Module):
    def __init__(
        self,
        is_grayscale: bool = True,
        height: int = 240,
        width: int = 360,
        use_resnet: bool = True,
        hidden_size: int = 20,
        num_lstm_layers: int = 2,
        num_diffusion_timesteps: int = 5,
        n_fft: int = 400,
    ):
        super(DiffusionVIG, self).__init__()
        self.cnn = modules.VideoCNN(
            hidden_size, use_resnet=use_resnet, is_grayscale=is_grayscale
        )
        self.lstm = modules.VideoLSTM(hidden_size, hidden_size, num_lstm_layers)
        self.audiowave = audiotransforms.InverseSpectrogram(n_fft=n_fft)
        self.num_timesteps = num_diffusion_timesteps

    def forward(self, x, _):
        """
        :param x: (batch_size, seq_len, height, width) video frames
        :param _: unused audio waveforms
        """
        # Fetch final hidden state after running video through cnn+lstm
        x = self.cnn(x)
        x = self.lstm(x)

        # Run through diffusion process
        spectrogram = self.diffusion_process(x)
        audio_waveform = self.synthesize_audiowave(spectrogram)
        print(audio_waveform.shape)
        sys.exit()
        return audio_waveform

    def diffusion_process(self, h: torch.Tensor, beta: float = 0.5):
        """
        :param h: (batch_size, self.hidden_size) set of hidden states for the batch
        """
        beta = [beta] * self.num_timesteps
        S = h

        # Forward diffusion
        for t in range(self.num_timesteps - 1, -1, -1):
            S += torch.randn_like(S) * math.sqrt(1 - beta[t])

        # Backward diffusion
        for t in range(self.num_timesteps):
            mu = torch.zeros_like(S)
            sigma = torch.ones_like(S)
            S -= mu + torch.randn_like(S) * sigma

        return S

    def synthesize_audiowave(self, spectrogram: torch.Tensor):
        # spectrogram shape: (batch_size, seq_len, dim)
        print(spectrogram.shape)
        spectrogram = spectrogram.permute(0, 2, 1)
        return self.audiowave(spectrogram)


if __name__ == "__main__":
    batch_size, seq_len, height, width = 2, 10, 24, 36
    hidden_size = 20
    model = DiffusionVIG(hidden_size=hidden_size)
    x = torch.rand([batch_size, seq_len, height, width])
    y = model(x, None)
    print(y.shape)
