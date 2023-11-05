import torch
import torch.nn as nn
import librosa
import numpy as np
import torch
import torch.nn as nn


class VIG(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VIG, self).__init__()
        self.alexnet = torch.hub.load("pytorch/vision", "alexnet", pretrained=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, _):
        cochleagrams = []

        x = self.alexnet(x)

        cochleagrams = self.generate_cochleagram(x)
        audio_waveform = self.inverse_cochleagram_to_audio(cochleagrams)
        audio_channel = audio_waveform[0, 0, :]
        audio_channel = audio_channel.reshape(-1)
        combined_audio_data = torch.cat([combined_audio_data, audio_channel])

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def generate_cochleagram(self, features):
        with torch.no_grad():
            features = features.view(1, 256, -1)
            cochleagram = self.lstm(features)
        return cochleagram.squeeze(1)

    def inverse_cochleagram_to_audio(self, cochleagram):
        inverted_audio = librosa.griffinlim(
            np.array(cochleagram), hop_length=512, win_length=256
        )
        return inverted_audio
