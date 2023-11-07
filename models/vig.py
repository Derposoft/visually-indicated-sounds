import cv2
import os
from glob import glob
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
import warnings
import torch.optim as optim
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import librosa
import torch.nn.functional as F

class CustomAlexnetFeatureExtractor(nn.Module):
    def __init__(self, input_channels=30):
        super(CustomAlexnetFeatureExtractor, self).__init__()

        # Load the AlexNet model without the final classification layers
        self.alexnet = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
        self.features = nn.Sequential(*list(self.alexnet.features.children())[:-2])  # Remove the last 2 layers

        # Replace the first convolutional layer with one that accepts input_channels
        self.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        return self.features(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # Add this line
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Get the batch size from the input tensor
        batch_size = x.size(0)

        # Add a time step dimension (sequence length = 1) to the input
        print("x.shape:", x.shape)
        x = x.unsqueeze(1)
        print("x.shape:", x.shape)

        # Initialize h0 and c0 as 2-D tensors
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through the LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)

        print("out.shape:", out.shape)
        return out


class vig(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, frame_rate=30):
        super(vig, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frame_rate = frame_rate

        # Initialize Alexnet feature extractor
        self.feature_extractor = CustomAlexnetFeatureExtractor(input_channels=30)

        self.lstm_model = LSTMModel(
        input_size = 9,
        hidden_size = 128,
        num_layers = 2,
        output_size = 256
        )

    def _initialize_feature_extractor(self):
        feature_extractor = models.alexnet(pretrained=True)
        # Remove the last classification layer to get features
        feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
        return feature_extractor

    def generate_cochleagram(self, features_all_frames):
        # Reshape the input features
        feature_array = features_all_frames.view(-1, self.input_size)

        # Convert features to cochleagram
        feature_array = feature_array.cpu().detach().numpy()
        cochleagram = librosa.feature.melspectrogram(S=feature_array.T)
        return cochleagram

    def forward(self, video_frames, audio_waves):
        # Extract video features with AlexNet
        video_features = self.feature_extractor(video_frames)

        print("video_features.shape:", video_features.shape)

        batch_size, seq_len, height, width = video_features.size()
        #video_features = video_features.view(batch_size, seq_len, height * width)
        video_features = video_features.view(batch_size * seq_len, height * width)

        print("batch_size:", batch_size)
        print("seq_len:", seq_len)
        print("height:", height)
        print("width:", width)
        print("**video_features.shape:", video_features.shape)

        # Forward pass through video LSTM
        video_output = self.lstm_model(video_features)

        return video_output
