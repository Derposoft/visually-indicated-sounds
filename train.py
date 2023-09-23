import torch.nn as nn
import argparse

from data.utils import load_data, download_data_if_not_downloaded


def train(model, train_dataloader, loss_function):
    for x, y in train_dataloader:
        preds = model(x)
        loss = loss_function(y, preds)
        loss.backwards()


if __name__ == "__main__":
    download_data_if_not_downloaded()
    train_dataloader, test_dataloader = load_data()
    model = None
    loss_function = None
    train(model, train_dataloader, loss_function)
