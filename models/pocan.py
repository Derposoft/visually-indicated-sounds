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
        self.spectrogram = audiotransforms.Spectrogram()
        self.audiowave = audiotransforms.InverseSpectrogram()

        # Ensure that "Default sound class" spectrograms are saved for future use
        # via this map from class -> spectrogram tensor
        self.default_spectrograms = create_default_spectrograms()

    def forward(self, x, _):
        """
        :param x: (batch_size, seq_len, height, width) video frames
        :param _: unused audio waveforms
        """
        x = self.cnn(x)
        self.c_hat_dist, self.s_additive = self.lstm(x)
        x = self.synthesize_spectrogram()
        x = self.audiowave(x)
        return x

    def synthesize_spectrogram(self):
        c_hat = torch.argmax(self.c_hat_dist, dim=-1)
        s_base = self.default_spectrograms[c_hat]
        s_additive = match_seq_len(s_base, self.s_additive)
        self.s_hat = s_base + s_additive
        return self.s_hat

    def loss(
        self,
        c: torch.Tensor,
        y: torch.Tensor,
        lbda: float = 1.0,
        mu: float = 1.0,
    ):
        """
        :param c: Ground truth sound classes labels of size (batch_size,)
        :param y: Ground truth waveform labels of size (batch_size, seq_len)
        """
        if not self.c_hat_dist:
            raise ValueError("Forward must be called first before calling loss!")

        # Classification loss
        loss_cls = torch.log(self.c_hat_dist[c])

        # Regression loss
        sp = self.s_hat
        s = self.spectrogram(y)
        loss_reg = self.smooth_l1_loss(s, sp)

        # Perceptual loss
        y_hat = self.audiowave(sp)
        loss_p = self.smooth_l1_loss(y, y_hat)

        return loss_cls + lbda * loss_reg + mu * loss_p
