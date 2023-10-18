import torch
import torch.nn as nn
import math
import torchvision.models as models


class VideoCNN(nn.Module):
    def __init__(self, output_size, use_resnet=False, is_grayscale=True):
        super(VideoCNN, self).__init__()
        self.grayscale_adapter = nn.Linear(1, 3) if is_grayscale else None
        default_num_classes = 1000
        if use_resnet:
            self.cnn = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1,
                num_classes=default_num_classes,
            )
        else:
            self.cnn = models.alexnet(
                weights=models.AlexNet_Weights.IMAGENET1K_V1,
                num_classes=default_num_classes,
            )
        self.fc = nn.Linear(default_num_classes, output_size)

    def forward(self, x):
        if self.grayscale_adapter:
            x = self.grayscale_adapter(x)

        x = self.cnn(x)
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
        # self.fc2_audio = nn.Linear(hidden_size, 1)
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
