"""
Implementation of of "Visually Indicated Sound Generation by Perceptually Optimized 
Classification".

https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Chen_Visually_Indicated_Sound_Generation_by_Perceptually_Optimized_Classification_ECCVW_2018_paper.pdf
"""
import torch
import torch.nn as nn
import torchvision.models as models

import math


def loss_cls(p, c):
    pass


def loss_reg(sp, s, lbda=1.0):
    pass


def loss_percep(y_hat, y, mu=1.0):
    pass


class SoundClassModel(nn.Module):
    def __init__(self) -> None:
        super(SoundClassModel, self).__init__()

    def loss_cls(p, c):
        pass


class VIGRnn(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        video_fps: float = 24,
        audio_sampling_rate_khz: float = 90,
        num_classes=None,
    ):
        """
        Creates an LSTM with a forward function which takes (batch_size, seq_len, input_size) tensors
        and outputs all hidden states as (batch_size, seq_len, hidden_size)

        :param input_size: dimension of each item in sequence
        :param hidden_size: hidden dimension size of lstm
        """
        super(VIGRnn, self).__init__()

        self.num_classes = num_classes
        # TODO for if we want to replicate hidden states from our cnn
        self.video_fps = video_fps
        self.audio_sampling_rate_khz = audio_sampling_rate_khz
        self.k = math.floor(self.audio_sampling_rate_khz / self.video_fps)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        audio_space_output_shape = (
            hidden_size + num_classes if num_classes else hidden_size
        )
        self.fc1_audio = nn.Linear(audio_space_output_shape)

    def forward(self, x: torch.Tensor):
        # Input size: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        x, h = self.lstm(x)
        h = self.fc1_audio(h)

        if self.num_classes:
            # TODO do something here to pull out the predicted class for POCAN
            pass

        return h


class POCAN(nn.Module):
    def __init__(self):
        super(POCAN, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        # self.fc1 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x, audio):
        # Remove the final classification layer of the CNN
        cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        dv = 1

        # Define the LSTM model
        lstm_hidden_size = 256  # Adjust as needed
        lstm_num_layers = 2  # Adjust as needed

        lstm = nn.LSTM(
            input_size=dv,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        x = self.fc1(x)
        return x
