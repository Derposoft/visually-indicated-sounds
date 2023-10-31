import torch
import torch.nn as nn
from torchvision import models, transforms

class AlexNetWithTransform(nn.Module):
    def __init__(self, num_classes=15):
        super(AlexNetWithTransform, self).__init__()

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Define the AlexNet model
        self.alexnet_model = models.alexnet(pretrained=False, num_classes=1000)  # Change num_classes to 1000

        # Modify the classifier part of AlexNet to match your number of output classes
        self.alexnet_model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.alexnet_model(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get the batch size from the input tensor
        batch_size = x.size(0)
        # Initialize h0 and c0 with the correct batch size
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # Adjust the view to match the batch size
        out, _ = self.lstm(x.view(batch_size, 1, -1), (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

class CombinedModel(nn.Module):
    def __init__(self, alexnet, lstm):
        super(CombinedModel, self).__init__()
        self.alexnet = alexnet
        self.lstm = lstm

    def forward(self, frames):
        # Extract features using AlexNet
        features = self.alexnet(frames)
        # Reshape features for LSTM input
        features = features.view(frames.size(0), -1)
        # Pass through LSTM
        output = self.lstm(features)
        return output

if __name__ == "__main__":

    num_classes = 15
    alexnet_model = AlexNetWithTransform(num_classes)
    #print(alexnet_model)

    input_size = 15  # The output feature size of AlexNet
    hidden_size = 15
    num_layers = 2
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    #print(lstm_model)

    combined_model = CombinedModel(alexnet_model, lstm_model)
    print(combined_model)
