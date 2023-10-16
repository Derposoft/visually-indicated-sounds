"""
Implementation of of "Visually Indicated Sound Generation by Perceptually Optimized 
Classification".

https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Chen_Visually_Indicated_Sound_Generation_by_Perceptually_Optimized_Classification_ECCVW_2018_paper.pdf
"""
import torch.nn as nn
import torchvision.models as models

from models.modules import VIGRnn


class SoundClassModel(nn.Module):
    def __init__(self) -> None:
        super(SoundClassModel, self).__init__()

    def loss_cls(p, c):
        pass


class POCAN(nn.Module):
    def __init__(self):
        super(POCAN, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        # self.fc1 = nn.Linear(in_features=64, out_features=10)
        self.lstm = VIGRnn()

    def forward(self, x, audio):
        # Remove the final classification layer of the CNN
        cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        dv = 1

        # Define the LSTM model
        lstm_hidden_size = 256  # Adjust as needed
        lstm_num_layers = 2  # Adjust as needed

        lstm = nn.LSTM(
            input_size=dv,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        x = self.fc1(x)
        return x
