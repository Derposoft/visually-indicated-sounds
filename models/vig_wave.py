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

# Feature Extraction with AlexNet
class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(AlexNetFeatureExtractor, self).__init__()
        self.alexnet = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
        self.features = self.alexnet.features

    def forward(self, x):
        return self.features(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # Remove unsqueeze(0) to maintain 3D input
        out = self.fc(out[:, -1, :])  # Take the last output and pass it through the linear layer
        return out

def generate_cochleagram(features, lstm_model):
    with torch.no_grad():
        features = features.view(1, 256, -1)
        #print(features.shape)
        cochleagram = lstm_model(features)
    return cochleagram.squeeze(1)

def inverse_cochleagram_to_audio(cochleagram):
    inverted_audio = librosa.griffinlim(np.array(cochleagram), hop_length=512, win_length=256)
    return inverted_audio

def preprocess_frame(frame):
    frame = cv2.resize(frame, (227, 227))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    return frame

def extract_frames(video_path, frame_rate):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
