import torch.nn as nn
import torch.optim as optim
import argparse

import data.utils as utils
from models.pocan import POCAN

import torch
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from models.preprocessdataforvig import load_and_preprocess_data
from models.vig_classification import AlexNetWithTransform, LSTMModel, CombinedModel
import torchvision.models as savemodel
import torch
import torch.nn as nn
from models.vig_wave import *


def evaluate_test_data(loaded_model, test_frames_array, test_labels_array, batch_size=32):

    batch_size, num_frames, height, width, num_channels = test_frames_array.shape
    test_frames_tensor = test_frames_array.reshape((batch_size * num_frames, num_channels, height, width))

    print("test_frames_array.shape:", test_frames_array.shape)
    print("test_frames_tensor.shape:", test_frames_tensor.shape)
    print("test_labels_array.shape:", test_labels_array.shape)

    # Convert data to PyTorch tensors
    test_frames_tensor = torch.tensor(test_frames_tensor, dtype=torch.float32)
    test_labels_array_new = [item for item in test_labels_array for _ in range(num_frames)]
    test_labels_array_new = torch.tensor(test_labels_array_new, dtype=torch.long)

    print("------")
    print("test_labels_array_new.shape:", test_labels_array_new.shape)
    print("test_frames_tensor.shape:", test_frames_tensor.shape)

    # Create a DataLoader for the test data
    test_data = TensorDataset(test_frames_tensor, test_labels_array_new)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Lists to store predictions and true labels for test data
    test_all_predictions = []
    test_all_labels = []

    for frames, labels in test_data_loader:
        # Perform inference using the loaded model
        frames = torch.tensor(frames, dtype=torch.float32)
        test_outputs = loaded_model(frames)

        # Convert model outputs to class predictions
        _, test_predicted = torch.max(test_outputs, 1)

        # Append the predictions and labels to the test lists
        test_all_predictions.extend(test_predicted.tolist())
        test_all_labels.extend(labels.tolist())

    # Calculate the confusion matrix for test data
    class_names = ['Bark', 'Cattle_bovinae', 'Bleat', 'rooster', 'Churchbell', 'Thunderstorm', 'Racecar_autoracing',
                       'Railtransport', 'Helicopter', 'Firealarm', 'Hammer', 'Gunshot_gunfire', 'Fireworks',
                       'Splash,splatter', 'Spray']
    test_confusion_matrix = confusion_matrix(test_all_labels, test_all_predictions)
    print(pd.DataFrame(test_confusion_matrix, columns=class_names))

    # Calculate the accuracy rate for test data
    test_accuracy = accuracy_score(test_all_labels, test_all_predictions)
    print(f"Test Accuracy Rate: {test_accuracy * 100:.2f}%")


def vig_train_and_get_best_model(combined_model, frames_array, labels_array, num_epochs=20, batch_size=32):
    best_accuracy = 0.0
    best_model_state = None

    batch_size, num_frames, height, width, num_channels = frames_array.shape
    frames_tensor = frames_array.reshape((batch_size * num_frames, 3, 224, 224))

    # Convert data to PyTorch tensors
    frames_tensor = torch.tensor(frames_tensor, dtype=torch.float32)
    labels_array_new = [item for item in labels_array for _ in range(15)]
    labels_array_new = torch.tensor(labels_array_new, dtype=torch.long)

    # Create a TensorDataset
    data = TensorDataset(frames_tensor, labels_array_new)

    # Create a DataLoader
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(combined_model.parameters(), lr=0.001, momentum=0.9)

    # Lists to store predictions and true labels
    class_names = ['Bark', 'Cattle_bovinae', 'Bleat', 'rooster', 'Churchbell', 'Thunderstorm', 'Racecar_autoracing',
                   'Railtransport', 'Helicopter', 'Firealarm', 'Hammer', 'Gunshot_gunfire', 'Fireworks',
                   'Splash,splatter', 'Spray']
    all_predictions = []
    all_labels = []

    # Training loop
    for epoch in range(num_epochs):
        for frames, labels in data_loader:
            optimizer.zero_grad()
            outputs = combined_model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Convert model outputs to class predictions
            _, predicted = torch.max(outputs, 1)

            # Append the predictions and labels to the lists
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

        # Calculate the confusion matrix
        print(pd.DataFrame(confusion_matrix(all_labels, all_predictions), columns=class_names))

        # Calculate the accuracy rate
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Accuracy Rate: {accuracy * 100:.2f}%")

        # Check if the current accuracy is better than the best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = combined_model

    return best_model_state, best_accuracy


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
        "--model", choices=["pocan", "foleygan", "vigclassification","vigwave"], type=str, required=True
    )
    parser.add_argument("--n_train", default=1000, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=1, type=int)  # testing value
    parser.add_argument("--frame_skip", default=10, type=int)
    parser.add_argument("--vid_height", default=240, type=int)
    parser.add_argument("--vid_width", default=360, type=int)
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
    elif config.model == "vigclassification":

        labels_file = 'models/train_labels.txt'
        video_directory = 'data/vig_train/'
        num_frames=15

        num_classes = 15
        input_size = 15  # The output feature size of AlexNet
        hidden_size = 15
        num_layers = 2

        alexnet_model = AlexNetWithTransform(num_classes)
        lstm_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        model = CombinedModel(alexnet_model, lstm_model)
        print(model)

        frames_array, labels_array = load_and_preprocess_data(labels_file, video_directory, num_frames)
        best_model_state, best_accuracy = vig_train_and_get_best_model(model, frames_array, labels_array, num_epochs=20, batch_size=32)
        print("best training accuracy:", best_accuracy)
        print(best_model_state)

        #test trained model
        test_labels_file = 'models/test_labels.txt'
        test_video_directory = 'data/vig_test/'

        test_frames_array, test_labels_array = load_and_preprocess_data(test_labels_file, test_video_directory, num_frames)
        evaluate_test_data(best_model_state, test_frames_array, test_labels_array, batch_size=32)

    elif config.model == "vigwave":

        combined_audio_data = np.array([])
        mp4_directory = 'data/vig_train/'
        mp4_files = glob(os.path.join(mp4_directory, '*.mp4'))

        # Training loop
        for video_path in mp4_files:
            print(video_path)

            frame_rate = 30
            frames = extract_frames(video_path, frame_rate)

            # Initialize AlexNet feature extractor
            feature_extractor = AlexNetFeatureExtractor()

            # Initialize LSTM model with the corrected input size
            input_size = 36  # Corrected input size to match the reshaped features (256x36)
            hidden_size = 512
            num_layers = 2
            output_size = 128

            lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
            # Process each frame and generate the cochleagram
            cochleagrams = []
            for frame in frames:
                frame = preprocess_frame(frame)
                features = feature_extractor(frame)

                cochleagram = generate_cochleagram(features, lstm_model)
                cochleagrams.append(cochleagram)

            # Inverse cochleagram transformation to audio
            audio_waveform = inverse_cochleagram_to_audio(cochleagrams)

            audio_channel = audio_waveform[0, 0, :]
            audio_channel = audio_channel.reshape(-1)
            combined_audio_data = np.append(combined_audio_data, audio_channel)

        # Initialize the LSTM model
        input_size = 1  # Adjust based on your data
        hidden_size = 64  # You can change this
        num_layers = 2  # You can change this
        num_classes = 10  # Adjust based on your task

        audio_lstm_model = AudioLSTM(input_size, hidden_size, num_layers, num_classes)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(audio_lstm_model.parameters(), lr=0.001)

        # Prepare your audio_channel data (combined_audio_data) for training
        audio_data = torch.from_numpy(combined_audio_data).float()

        # Convert the data into input format (sequence_length, input_size)
        sequence_length = len(audio_data)
        input_data = audio_data.view(1, sequence_length, input_size)

        # Training loop
        num_epochs = 20
        for epoch in range(num_epochs):
            audio_lstm_model.train()
            optimizer.zero_grad()

            outputs = audio_lstm_model(input_data)
            # Provide your ground truth labels (replace 'labels' with your actual labels)
            labels = torch.randint(0, num_classes, (1,))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}')

        # Save the trained audio LSTM model
        torch.save(audio_lstm_model.state_dict(), 'audio_lstm_model.pth')

    if (config.model == "vigwave") or (config.model == "vigclassification"):
        print("Already trained")
    else:
        loss_function = nn.CrossEntropyLoss()
        opt = optim.SGD(model.parameters(), lr=0.01)
        train(
            model, train_dataloader, loss_function, opt, num_epochs=epochs, verbose=verbose
        )
