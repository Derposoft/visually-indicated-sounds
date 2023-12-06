import torch
import torch.nn as nn
import math
import torchvision.models as models

from data.utils import match_seq_len


class VideoCNN(nn.Module):
    def __init__(self, output_size, use_resnet=False, is_grayscale=True):
        super(VideoCNN, self).__init__()
        self.grayscale_adapter = nn.Linear(1, 3) if is_grayscale else None
        default_num_classes = 1000
        if use_resnet:
            self.cnn = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1,
                num_classes=default_num_classes,
            ).eval()
        else:
            self.cnn = models.alexnet(
                weights=models.AlexNet_Weights.IMAGENET1K_V1,
                num_classes=default_num_classes,
            ).eval()
        self.fc = nn.Linear(default_num_classes, output_size)

    def forward(self, x: torch.Tensor):
        """
        :param x: Video input. Input size expectation: (batch_size, n_frames, height, width, [depth]), with
        depth dimension only present if is_grayscale=False during initialization.
        """
        if self.grayscale_adapter:
            batch_size, n_frames, height, width = x.shape
            x = x.reshape([batch_size, n_frames, height, width, 1])
            x = self.grayscale_adapter(x)

        # Update indices to (batch_size, n_frames, depth, height, width) and batch all frames
        x = x.permute(0, 1, 4, 2, 3)
        batch_size, n_frames, depth, height, width = x.shape
        x = x.reshape([-1, depth, height, width])
        x = self.cnn(x)

        # Separate batch and frames again and project to output size
        x = x.reshape([batch_size, n_frames, -1])
        x = self.fc(x)
        return x


class VideoLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        video_fps: float = 24,
        audio_sampling_rate_khz: float = 90,
        num_classes=None,
        predict_class=False,
    ):
        """
        Creates an LSTM with a forward function which takes (batch_size, seq_len, input_size) tensors
        and outputs all hidden states as (batch_size, seq_len, hidden_size)

        :param input_size: dimension of each item in sequence
        :param hidden_size: hidden dimension size of lstm
        """
        super(VideoLSTM, self).__init__()

        assert (
            not predict_class or num_classes != None
        ), "num_classes must be set if we're predicting classes."
        self.predict_class = predict_class
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.video_fps = video_fps
        self.audio_sampling_rate_khz = audio_sampling_rate_khz
        self.k = math.floor(self.audio_sampling_rate_khz / self.video_fps)  # TODO

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        audio_space_output_shape = (
            hidden_size + num_classes if num_classes else hidden_size
        )
        self.fc1_audio = nn.Linear(hidden_size, audio_space_output_shape)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        # Input size: (batch_size, seq_len, dim)
        assert len(x.shape) == 3, "Must have batch size > 1"
        batch_size, seq_len, dim = x.shape
        x, _ = self.lstm(x)
        x = self.fc1_audio(x)

        # If using LSTM to predict class
        if self.predict_class:
            c, x = x[:, :, : self.num_classes], x[:, :, self.num_classes :]
            c = self.softmax(c)
            c = torch.sum(c, dim=1) / seq_len
            return c, x

        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, image_height, color_channels=1):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(input_size, image_height * 8, 2, 1, 0, bias=False),
            nn.BatchNorm1d(image_height * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose1d(image_height * 8, image_height * 4, 2, 2, 1, bias=False),
            nn.BatchNorm1d(image_height * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose1d(image_height * 4, image_height * 2, 2, 2, 1, bias=False),
            nn.BatchNorm1d(image_height * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose1d(image_height * 2, image_height, 2, 2, 1, bias=False),
            nn.BatchNorm1d(image_height),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose1d(image_height, color_channels, 2, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = x.reshape(batch_size, -1)
        x = self.linear(x)
        return x


def calculate_audiowave_loss(y: torch.Tensor, y_preds: torch.Tensor) -> torch.Tensor:
    """
    :param y: audiowaves batch of shape (batch_size, audiowave_len)
    :param y_pred: audiowave predictions batch of shape (batch_size, audiowave_preds_len)
    :returns: The loss between the two audiowaves as a tensor
    """
    y_preds = match_seq_len(
        torch.zeros([y.shape[1], 1]),
        y_preds.reshape([y_preds.shape[0], y_preds.shape[1], 1]),
    ).reshape(
        [y_preds.shape[0], -1]
    )  # match_seq_len was poorly thought out. oops.
    loss = nn.SmoothL1Loss()(y, y_preds)
    return torch.abs(loss)


if __name__ == "__main__":
    # Test that the LSTM works as expected for now. Should receive an input
    # of shape (batch_size, sequence_len, input_dim) and output a size of (batch_size, sequence_len, 1),
    # which represents the mono audio output for that input.
    batch_size = 10
    seq_len = 100
    input_size = 1000
    lstm_hidden_size = 20
    lstm_layers = 2
    rnn = VideoLSTM(input_size, lstm_hidden_size, lstm_layers)
    test_input = torch.rand([batch_size, seq_len, input_size])
    test_output = rnn(test_input)
    received_output_size = test_output.shape
    expected_output_size = torch.Size([batch_size, seq_len, 1])
    assert received_output_size == expected_output_size
