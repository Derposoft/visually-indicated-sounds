import torch
import torch.nn as nn
import torchaudio.transforms as audiotransforms

from models.modules import VideoCNN, VideoLSTM, calculate_audiowave_loss
from data.utils import match_seq_len
import torch.nn.functional as F

class VIG(nn.Module):
    def __init__(
        self, hidden_size, num_layers, is_grayscale=False, frame_rate=30, n_fft=400
    ):
        super(VIG, self).__init__()

        # calculate lstm output size
        if n_fft % 2 == 1:
            n_fft += 1
        lstm_hidden_size = int(n_fft / 2) + 1
        lstm_output_size = 2 * lstm_hidden_size  # x2 since we fold it as complex tensor

        self.cnn = VideoCNN(hidden_size, use_resnet=False, is_grayscale=is_grayscale)
        self.lstm = VideoLSTM(
            hidden_size, lstm_output_size, num_layers, video_fps=frame_rate
        )
        self.audiowave = audiotransforms.InverseSpectrogram(n_fft=n_fft)
        self.loss_function = nn.SmoothL1Loss()

    def inverse_cochleagram_to_audio(self, cochleagram: torch.Tensor):
        # cochleagram shape: (batch_size, seq_len, hidden_size)
        cochleagram = cochleagram.permute(0, 2, 1)
        return self.audiowave(cochleagram)

    def forward(self, x: torch.Tensor, _):
        x = self.cnn(x)  # -> (batch_size, seq_len, hidden_size)
        x = self.lstm(x)  # -> (batch_size, seq_len, 2*hidden_size)

        # "fold" x into (batch_size, seq_len, hidden_size) so we get real and imaginary parts
        x_real, x_imag = x.chunk(2, dim=-1)
        x = torch.complex(x_real, x_imag)
        x = self.inverse_cochleagram_to_audio(x)
        return x

    def loss(self, outputs: torch.Tensor, _: torch.Tensor, audiowaves: torch.Tensor):

        #print("outputs.shape:", outputs.shape)
        #print("audiowaves.shape:", audiowaves.shape)

        target_size = outputs.shape[1]
        audiowaves_downsampled = F.interpolate(audiowaves.unsqueeze(1), size=target_size, mode='linear', align_corners=False)
        audiowaves_downsampled = audiowaves_downsampled.squeeze(1)

        #print("downsampled audiowaves.shape:", audiowaves_downsampled.shape)
        audiowaves = audiowaves_downsampled

        loss = nn.MSELoss()
        output = loss(audiowaves, outputs)

        return output
