import torch
import torch.nn as nn
import math


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
        self.fc1_audio = nn.Linear(hidden_size, audio_space_output_shape)

    def forward(self, x: torch.Tensor):
        # Input size: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        x, h = self.lstm(x)
        h = self.fc1_audio(h)

        if self.num_classes:
            # TODO do something here to pull out the predicted class for POCAN
            pass

        return h
