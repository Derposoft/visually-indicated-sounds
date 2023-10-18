import torch.nn as nn
import torch.optim as optim
import argparse

import data.utils as utils
from models.pocan import POCAN


def train(model, train_dataloader, criterion, opt, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for video_frames, audio_waves, labels in train_dataloader:
            opt.zero_grad()
            outputs = model(video_frames, audio_waves)

            if isinstance(model, POCAN):
                # POCAN custom loss
                loss = model.loss(labels, audio_waves)
            else:
                # Default loss
                loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.4f}")


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["pocan", "foleygan", "vig"], type=str, required=True
    )
    parser.add_argument("--n_train", default=1000, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--batch_size", default=1, type=int)  # testing value
    parser.add_argument("--frame_skip", default=10, type=int)
    parser.add_argument("--vid_height", default=240, type=int)
    parser.add_argument("--vid_width", default=360, type=int)
    parser.add_argument("--grayscale", default=True, type=bool)
    config = parser.parse_args()
    n_train = config.n_train
    n_test = config.n_test
    batch_size = config.batch_size
    frame_skip = config.frame_skip
    vid_height = config.vid_height
    vid_width = config.vid_width
    grayscale = config.grayscale

    # Download data and get dataloaders
    utils.download_data_if_not_downloaded(n_train_videos=n_train, n_test_videos=n_test)
    train_dataloader, test_dataloader = utils.load_data(
        model=config.model,
        batch_size=batch_size,
        vid_height=vid_height,
        vid_width=vid_width,
        frame_skip=frame_skip,
        grayscale=grayscale,
    )

    # Train models
    annotations, class_map = utils.load_annotations_and_classmap()
    num_classes = len(class_map)
    if config.model == "foleygan":
        model = None  # TODO
    elif config.model == "pocan":
        hidden_size = 20
        num_lstm_layers = 2
        model = POCAN(
            num_classes,
            is_grayscale=grayscale,
            height=vid_height,
            width=vid_width,
            use_resnet=False,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
        )
    elif config.model == "vig":
        model = None  # TODO

    assert model != None
    loss_function = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01)
    train(model, train_dataloader, loss_function, opt)
