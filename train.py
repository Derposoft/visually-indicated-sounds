import argparse
import torch.nn as nn
import torch.optim as optim

import data.utils as utils
from models.pocan import POCAN
from models.vig import VIG
from models.foleygan import foleygan


def train(model, train_dataloader, criterion, opt, num_epochs=10, verbose=False):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for video_frames, audio_waves, labels in train_dataloader:
            opt.zero_grad()
            outputs = model(video_frames, audio_waves)

            # Custom losses by model
            if isinstance(model, POCAN):
                loss = model.loss(labels, audio_waves)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

            if verbose:
                print(f"Current running loss: {running_loss}")

        average_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.4f}")


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["pocan", "foleygan", "vig"],
        type=str,
        required=True,
    )
    parser.add_argument("--n_train", default=10, type=int)
    parser.add_argument("--n_test", default=5, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=1, type=int)  # testing value
    parser.add_argument("--frame_skip", default=10, type=int)
    parser.add_argument("--vid_height", default=64, type=int)
    parser.add_argument("--vid_width", default=64, type=int)
    parser.add_argument("--no_grayscale", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    config = parser.parse_args()
    n_train = config.n_train
    n_test = config.n_test
    batch_size = config.batch_size
    frame_skip = config.frame_skip
    vid_height = config.vid_height
    vid_width = config.vid_width
    grayscale = not config.no_grayscale
    verbose = config.verbose
    epochs = config.epochs

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
    annotations, class_map = utils.load_annotations_and_classmap()
    num_classes = len(class_map)

    # Create models
    if config.model == "foleygan":
        img_feature_dim = 64
        hidden_size = 20
        n_fft = num_classes
        model = foleygan(img_feature_dim, num_classes, hidden_size, n_fft)
        loss_function = nn.HingeEmbeddingLoss()
        opt = optim.Adam(model.parameters(), lr=0.0001)
    elif config.model == "pocan":
        hidden_size = 5
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
        loss_function = nn.CrossEntropyLoss()
        opt = optim.SGD(model.parameters(), lr=0.01)
    elif config.model == "vig":
        hidden_size = 64
        num_layers = 2
        model = VIG(hidden_size, num_layers, is_grayscale=grayscale)
        loss_function = nn.CrossEntropyLoss()
        opt = optim.SGD(model.parameters(), lr=0.01)

    assert model != None

    # Train models
    train(
        model,
        train_dataloader,
        loss_function,
        opt,
        num_epochs=epochs,
        verbose=verbose,
    )
