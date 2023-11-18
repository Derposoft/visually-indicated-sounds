import argparse
import torch.nn as nn
import torch.optim as optim

import data.utils as utils
from models.pocan import POCAN
from models.vig import VIG
from models.foleygan import FoleyGAN
from models.modules import calculate_audiowave_loss


def train(model, train_dataloader, test_dataloader, opt, num_epochs=10, verbose=False):
    """
    Trains the given model. Assumes that model is an nn.Module class, with a function
    defined inside of it called "loss". Loss functions in models must look like:

    def loss(outputs, labels, audio)

    where outputs are model outputs, labels are sound class labels, and audio is the raw
    audio for each video.
    """
    for epoch in range(num_epochs):
        running_loss = 0.0
        for video_frames, audio, audio_raw, labels in train_dataloader:
            opt.zero_grad()
            outputs = model(video_frames, audio)

            # Custom losses by model
            loss = model.loss(outputs, labels, audio_raw)
            loss.backward()
            opt.step()
            running_loss += loss.item()

            if verbose:
                print(f"Current running loss: {running_loss}")
            
        test(model, test_dataloader)

        average_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss}")

def test(model, test_dataloader):
    """
    Tests the given model.
    """
    total_mse = 0
    for video_frames, audio, audio_raw, labels in test_dataloader:
        outputs = model(video_frames, audio)
        total_mse += calculate_audiowave_loss(audio_raw, outputs)
    
    average_mse = total_mse/len(test_dataloader)
    
    print(f"Total MSE: [{total_mse}]; Average MSE: [{average_mse}]")



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
    parser.add_argument("--batch_size", default=1, type=int)  # test val
    parser.add_argument("--lr", default=1e-3, type=float)
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
        n_fft = 400
        model = FoleyGAN(img_feature_dim, num_classes, hidden_size, batch_size, n_fft)
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
    elif config.model == "vig":
        hidden_size = 64
        num_layers = 2
        model = VIG(hidden_size, num_layers, is_grayscale=grayscale)

    assert model != None
    opt = optim.Adam(model.parameters(), lr=config.lr)

    # Train models
    train(
        model,
        train_dataloader,
        test_dataloader,
        opt,
        num_epochs=epochs,
        verbose=verbose,
    )
