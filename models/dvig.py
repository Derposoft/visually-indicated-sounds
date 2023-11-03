"""
Implementation of of D-VIG as described in our midterm report.
"""
import torch
import torch.nn as nn

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
    ):
        super(DiffusionVIG, self).__init__()
        self.cnn = modules.VideoCNN(hidden_size, use_resnet=use_resnet, is_grayscale=is_grayscale)
        self.lstm = modules.VideoLSTM(hidden_size, hidden_size, num_lstm_layers)
        self.num_timesteps = num_diffusion_timesteps

    def forward(self, x, _):
        """
        :param x: (batch_size, seq_len, height, width) video frames
        :param _: unused audio waveforms
        """
        # Fetch final hidden state after running video through cnn+lstm
        x = self.cnn(x)
        x = self.lstm(x)
        x = x[:, -1]

        # Run through diffusion process
        spectrogram = self.diffusion_process(x)
        audio_waveform = self.synthesize_audiowave(spectrogram)
        return audio_waveform

    def diffusion_process(self, h):
        """
        :param h: (batch_size, self.hidden_size) set of hidden states for the batch
        """
        # Forward diffusion
        S = h  # Initialize S with the hidden state
        for t in range(self.num_timesteps - 1, -1, -1):
            S_t = S + torch.randn_like(S) * torch.sqrt(1 - self.beta[t])  # Sample S_t
            S = S_t  # Update S

        # Backward diffusion
        for t in range(self.num_timesteps):
            mu = torch.zeros_like(S)
            sigma = torch.ones_like(S)
            S_t = mu + torch.randn_like(S) * sigma  # Sample S_t
            S = S_t  # Update S

        return S  # The final sample

    def synthesize_audiowave(self, spectrogram: torch.Tensor):
        # spectrogram shape: (batch_size, seq_len, dim)
        spectrogram = spectrogram.permute(0, 2, 1)
        return self.audiowave(spectrogram)
