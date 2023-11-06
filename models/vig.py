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


class CustomFeatureExtractor(nn.Module):
    def __init__(self, input_channels=30):
        super(CustomFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # Add more layers as needed

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Forward pass through additional layers
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)

        # Check the shape of the out tensor
        if out.ndim == 3:
            # If it's a 3D tensor, extract the last output
            out = self.fc(out[:, -1, :])
        elif out.ndim == 2:
            # If it's a 2D tensor, add a time step dimension and then extract the last output
            out = self.fc(out[:, -1, :].unsqueeze(1))
        else:
            raise ValueError("Unexpected input tensor shape")

        return out

class vig(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, frame_rate=30):
        super(vig, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frame_rate = frame_rate

        # Initialize custom feature extractor
        self.feature_extractor = CustomFeatureExtractor(input_channels=30)


        # Initialize LSTM model
        self.lstm_model = self._initialize_lstm_model()

    def _initialize_feature_extractor(self):
        feature_extractor = models.alexnet(pretrained=True)
        # Remove the last classification layer to get features
        feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])
        return feature_extractor

    def _initialize_lstm_model(self):
        lstm_model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.num_classes)
        return lstm_model

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

        print(video_features.shape)
        batch_size, seq_len, height, width = video_features.size()
        video_features = video_features.view(batch_size, seq_len, height * width)

        # Forward pass through video LSTM
        video_output = self.lstm_model(video_features)

        # Forward pass audio_waves through audio LSTM or other audio model if needed
        audio_output = self.audio_model(audio_waves)

        # Combine the video and audio features as needed
        combined_features = self.combine_features(video_output, audio_output)

        return combined_features

    def audio_model(self, audio_waves):
        # Implement your audio feature extraction or model here
        # This could be another LSTM or any other model for audio data
        pass

    def combine_features(self, video_features, audio_features):
        # Implement how you want to combine video and audio features
        # This can be a simple concatenation, element-wise addition, etc.
        pass


"""
class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(AlexNetFeatureExtractor, self).__init__()
        self.alexnet = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
        self.features = self.alexnet.features

    def forward(self, x):
        return self.features(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)

        # Check the shape of the out tensor
        if out.ndim == 3:
            # If it's a 3D tensor, extract the last output
            out = self.fc(out[:, -1, :])
        elif out.ndim == 2:
            # If it's a 2D tensor, add a time step dimension and then extract the last output
            out = self.fc(out[:, -1, :].unsqueeze(1))
        else:
            raise ValueError("Unexpected input tensor shape")

        return out

def generate_cochleagram(features_all_frames):
    # Reshape the input features
    feature_array = features_all_frames.view(-1, 9216)  # Assuming the shape is (time frames, features)

    # Convert features to cochleagram
    feature_array = feature_array.cpu().detach().numpy()  # Convert to NumPy array
    cochleagram = librosa.feature.melspectrogram(S=feature_array.T)  # Transpose for the correct shape

    return cochleagram
"""
