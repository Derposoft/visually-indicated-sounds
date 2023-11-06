import argparse
import torch.nn as nn
import torch.optim as optim

import data.utils as utils
from models.pocan import POCAN
from models.vig import vig
from models.foleygan import foleygan

def vig_preprocess_frame(frame):
    frame = cv2.resize(frame, (227, 227))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    return frame

def vig_extract_frames(video_path, frame_rate):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def vig_train(lstm_model, num_epochs=10, frame_rate=30):

    combined_audio_data = np.array([])
    mp4_directory = 'data/vig_train/'
    mp4_files = glob(os.path.join(mp4_directory, '*.mp4'))

    count = 0
    # Training loop
    for video_path in mp4_files:
        print(video_path)

        frames = vig_extract_frames(video_path, frame_rate)

        # Initialize AlexNet feature extractor
        feature_extractor = AlexNetFeatureExtractor()

        # Process each frame and generate the cochleagram
        features_all_frames = []
        print("frames.shape:", len(frames))
        for frame in frames:
            frame = vig_preprocess_frame(frame)
            features = feature_extractor(frame)
            features_all_frames.append(features)

        # Train LSTM
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

        # Convert the list of features into a single tensor
        print("len(features_all_frames):", len(features_all_frames))
        print("len(features_all_frames[0]):", len(features_all_frames[0]))

        features_all_frames = torch.stack(features_all_frames)
        best_model = None

        for epoch in range(num_epochs):
            lstm_model.train()
            optimizer.zero_grad()

            # Pass features of all frames through LSTM
            print("features_all_frames.shape:", features_all_frames.shape)
            # Reshape features_all_frames to match the expected input size
            features_all_frames = features_all_frames.view(1, -1, input_size)  # Assuming batch size of 1
            print("features_all_frames.shape:", features_all_frames.shape)
            outputs = lstm_model(features_all_frames)

            output_size = torch.Size([1, 128])
            num_classes = 15  # Number of classes
            labels = torch.randint(0, num_classes, output_size)

            print("outputs.shape:", outputs.shape)
            print("labels.shape:", labels.shape)

            outputs = outputs.to(torch.float)
            labels = labels.to(torch.float)

            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


        print("features_all_frames.shape:", features_all_frames.shape)
        cochleagram = generate_cochleagram(features_all_frames)
        print("cochleagram:", cochleagram)

        if count == 1:
            break
        else:
            count = count + 1


def train(model, train_dataloader, criterion, opt, num_epochs=10, verbose=False):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for video_frames, audio_waves, labels in train_dataloader:
            opt.zero_grad()

            print("video_frames.shape:", video_frames.shape)
            print("audio_waves.shape:", audio_waves.shape)

            outputs = model(video_frames, audio_waves)

            print("outputs.shape:", outputs.shape)
            print("outputs:", outputs)

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
        #input_size = 9216 #initial
        #hidden_size = 512 #initial
        #num_layers = 2 #initial
        #output_size = 128 #initial

        #model = LSTMModel(input_size, hidden_size, num_layers, output_size) #initial

        input_size = 256
        hidden_size = 512
        num_layers = 2
        output_size = 128

        model = vig(num_classes, input_size, hidden_size, num_layers)

        print(model)

        loss_function = nn.CrossEntropyLoss()
        opt = optim.SGD(model.parameters(), lr=0.01)

    assert model != None

    # Train models
    if config.model != "vig":
        train(
            model, train_dataloader, loss_function, opt, num_epochs=epochs, verbose=verbose
        )
    else:
        train(
            model, train_dataloader, loss_function, opt, num_epochs=10, verbose=verbose
        )
        #vig_train(model, num_epochs=10, frame_rate=30) #initial
