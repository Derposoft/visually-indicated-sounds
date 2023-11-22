"""
Implementation of of "Visually Indicated Sound Generation by Perceptually Optimized 
Classification".

https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Chen_Visually_Indicated_Sound_Generation_by_Perceptually_Optimized_Classification_ECCVW_2018_paper.pdf
"""
import torch
import torch.nn as nn
import torchaudio.transforms as audiotransforms

import models.modules as modules
from data.utils import create_default_spectrograms, match_seq_len


class POCAN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        is_grayscale: bool = True,
        height: int = 240,
        width: int = 360,
        use_resnet: bool = False,
        hidden_size: int = 20,
        num_lstm_layers: int = 2,
    ):
        super(POCAN, self).__init__()
        cnn_output_dim = 2048 if use_resnet else 4096
        self.cnn = modules.VideoCNN(
            output_size=cnn_output_dim, use_resnet=False, is_grayscale=is_grayscale
        )
        self.lstm = modules.VideoLSTM(
            cnn_output_dim,
            hidden_size,
            num_lstm_layers,
            num_classes=num_classes,
            predict_class=True,
        )
        self.s_additive = None
        self.c_hat_dist = None
        self.s_hat = None
        self.smooth_l1_loss = nn.SmoothL1Loss()
        n_fft = (hidden_size - 1) * 2
        self.spectrogram = audiotransforms.Spectrogram(n_fft=n_fft)
        self.audiowave = audiotransforms.InverseSpectrogram(n_fft=n_fft)

        # Ensure that "Default sound class" spectrograms are saved for future use
        # via this map from class -> spectrogram tensor
        self.default_spectrograms = create_default_spectrograms(n_fft=n_fft)

    def forward(self, x, _):
        """
        :param x: (batch_size, seq_len, height, width) video frames
        :param _: unused audio waveforms
        """
        x = self.cnn(x)
        self.c_hat_dist, self.s_additive = self.lstm(x)
        x = self.synthesize_spectrogram()
        x = self.synthesize_audiowave(x)
        return x

    def synthesize_audiowave(self, spectrogram: torch.Tensor):
        # spectrogram shape: (batch_size, seq_len, dim)
        spectrogram = spectrogram.permute(0, 2, 1)
        return self.audiowave(spectrogram)

    def synthesize_spectrogram(self):
        c_hat = torch.argmax(self.c_hat_dist, dim=-1)
        s_base = [self.default_spectrograms[_c_hat.item()] for _c_hat in c_hat]
        s_base = torch.nn.utils.rnn.pad_sequence(s_base, batch_first=True)
        s_additive = match_seq_len(s_base, self.s_additive, tgt_is_3d=True)
        self.s_hat = s_base + s_additive
        return self.s_hat

    def loss(
        self,
        _: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        lbda: float = 50.0,
        mu: float = 100.0,
    ):
        """
        :param c: Ground truth sound classes labels of size (batch_size,)
        :param y: Ground truth waveform labels of size (batch_size, seq_len)
        """
        if self.c_hat_dist is None:
            raise ValueError("Forward must be called first before calling loss!")

        # Classification loss
        loss_cls = -torch.sum(
            torch.log(self.c_hat_dist[range(len(self.c_hat_dist)), c - 1])
        )

        # Regression loss
        s = self.spectrogram(y).permute(0, 2, 1)  # (batch, seq, dim)
        sp = match_seq_len(s, self.s_hat, tgt_is_3d=True)
        s_mag, sp_mag = torch.abs(s), torch.abs(sp)  # hack? not sure what paper did
        loss_reg = self.smooth_l1_loss(s_mag, sp_mag)
        loss_reg = torch.abs(loss_reg)  # do i even need this?

        # Perceptual loss
        y_hat = self.synthesize_audiowave(sp)
        loss_p = modules.calculate_audiowave_loss(y, y_hat)

        return loss_cls + lbda * loss_reg + mu * loss_p
