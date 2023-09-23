import torch.nn as nn
import torch.optim as optim
import argparse

from data.utils import load_data, download_data_if_not_downloaded
from models.pocan import POCAN


def train(model, train_dataloader, criterion, opt, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.4f}")


if __name__ == "__main__":
    download_data_if_not_downloaded()
    train_dataloader, test_dataloader = load_data()
    model = POCAN()
    loss_function = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01)
    train(model, train_dataloader, loss_function, opt)
