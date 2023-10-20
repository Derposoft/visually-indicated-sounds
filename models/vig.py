import torch
import torch.nn as nn
from torchvision import models, transforms

class cnn_lstm_model(nn.Module):
    def __init__(self, num_classes, input_size=4096, hidden_size=512, num_layers=2):
        super(CombinedModel, self).__init__()

        # Define the transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Define the AlexNet model
        self.alexnet = models.alexnet(pretrained=False, num_classes=15)

        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)

        # Initialize the LSTM model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, frames):
        # Apply transformations to frames
        frames = self.transform(frames)

        # Extract features using AlexNet
        features = self.alexnet(frames)

        # Reshape features for LSTM input
        features = features.view(frames.size(0), -1)

        # Pass through LSTM
        h0 = torch.zeros(self.num_layers, features.size(0), self.hidden_size).to(features.device)
        c0 = torch.zeros(self.num_layers, features.size(0), self.hidden_size).to(features.device)
        lstm_out, _ = self.lstm(features, (h0, c0))

        # Take the output of the last time step and pass it through a fully connected layer
        lstm_out = self.fc(lstm_out[:, -1, :])

        return lstm_out

#num_classes = 15  # Change this to your desired number of classes
#combined_model = CombinedModel(num_classes)
#print(combined_model)  # Print the model architecture
