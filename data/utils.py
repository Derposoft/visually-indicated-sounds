import argparse
import cv2
from functools import lru_cache
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import os
import pickle as pkl
from pytube import YouTube
import scipy.signal as signal
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as audiotransforms
from tqdm import tqdm


def get_filename_from_video_id(video_id: str):
    return f"{video_id}.mp4"


def download_video(video_id: str, dirname: str):
    # URL of the YouTube video you want to download
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(video_url)
    filename = get_filename_from_video_id(video_id)
    video_stream = yt.streams.get_lowest_resolution()
    video_stream.download(output_path=dirname, filename=filename)
    return filename


def download_data(train_dir: str, test_dir: str, num_train: int = 5, num_test: int = 2):
    # Read video id data file
    max_videos = num_train + num_test
    video_id_file = os.path.join(os.path.dirname(__file__), "vig_dl.lst")
    annotations_dict, _ = load_annotations_and_classmap()
    with open(video_id_file) as fin:
        lines = fin.readlines()

    for video_idx, line in tqdm(enumerate(lines[:max_videos])):
        youtube_video_id, video_id = line.strip().split(",")
        video_id = int(video_id)
        download_dir = train_dir if video_idx < num_train else test_dir

        # Download video
        try:
            download_video_filename = download_video(youtube_video_id, download_dir)
        except:
            continue

        # Clip video with ffmpeg if it is a snippet
        download_video_saved_path = os.path.join(download_dir, download_video_filename)
        clipped_video_saved_path = os.path.join(
            download_dir, "clipped_" + download_video_filename
        )
        if video_id in annotations_dict:
            annotation = annotations_dict[video_id]
            start_label, end_label = "start_time", "end_time"
            if start_label not in annotation or end_label not in annotation:
                continue
            start_time = int(annotation[start_label])
            end_time = int(annotation[end_label])
            ffmpeg_extract_subclip(
                download_video_saved_path,
                start_time,
                end_time,
                targetname=clipped_video_saved_path,
            )

            # Replace original video with clipped video
            os.remove(download_video_saved_path)
            shutil.move(clipped_video_saved_path, download_video_saved_path)


def download_data_if_not_downloaded(n_train_videos=1000, n_test_videos=200):
    train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
    test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
    data_files = [f for f in os.listdir(train_dir) if f.endswith(".mp4")]
    if not data_files:
        download_data(train_dir, test_dir, n_train_videos, n_test_videos)


@lru_cache(maxsize=None)
def load_video_ids():
    map = {}
    with open(os.path.join(os.path.dirname(__file__), "vig_dl.lst")) as f:
        lines = f.readlines()
    for line in lines:
        video_id, dataset_id = line.split(",")
        map[video_id] = dataset_id
    return map


def get_audio_features_by_video_id(video_id: str, data_dir: str) -> torch.Tensor:
    """
    Perform audio feature generation as described by "Visually Indicated Sounds", Owens et al.

    Steps:
    1. Apply band-pass filters w/ ERB spacing
    2. Compute subband envelopes
    3. Downsample and compress envelopes
    """

    # Get waveform from video_id, set algorithm parameters
    file_path = os.path.join(data_dir, get_filename_from_video_id(video_id))
    waveform, sample_rate = torchaudio.load(file_path)
    out_rate = 90
    compression_constant = 0.3

    # Apply filter, compute envelope. Here we choose mel filterbank
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
    spectrogram = mel_spectrogram(waveform)
    spectrogram_tensor = torch.log1p(spectrogram)
    spectrogram = spectrogram_tensor.numpy()
    envelope = signal.hilbert(spectrogram, axis=0)
    envelope = torch.Tensor(envelope)

    # Downsample and compress the envelopes
    downsampling = audiotransforms.Resample(orig_freq=sample_rate, new_freq=out_rate)
    downsampled_envelopes = downsampling(envelope)
    compressed_envelopes = torch.abs(downsampled_envelopes) ** compression_constant
    return compressed_envelopes


def video_id_to_dataset_id(video_id: str) -> int:
    map = load_video_ids()
    return int(map[video_id])


def load_annotations_and_classmap():
    annotation_file = "vig_annotation.pkl"
    annotation_file = os.path.join(os.path.dirname(__file__), annotation_file)
    class_map_file = "vig_class_map.pkl"
    class_map_file = os.path.join(os.path.dirname(__file__), class_map_file)

    with open(annotation_file, "rb") as f:
        annotations = pkl.load(f, encoding="latin-1")

    with open(class_map_file, "rb") as f:
        class_map = pkl.load(f)
    return annotations, class_map


class VideoDataset(Dataset):
    def __init__(
        self,
        data_dir=os.path.join(os.path.dirname(__file__), "./vig"),
        transform=None,
        frame_skip=10,
    ):
        self.data_dir = data_dir
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
        self.transform = transform
        self.frame_skip = frame_skip
        self.annotations, self.class_map = load_annotations_and_classmap()

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # Read video frames
        video_id = self.video_files[idx].split(".")[0]
        video_file = os.path.join(self.data_dir, self.video_files[idx])
        cap = cv2.VideoCapture(video_file)
        frames = []
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % self.frame_skip == 0:
                frames.append(frame)
            frame_number += 1
        cap.release()

        # Apply frame size normalization transformation
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        video_frames_tensor = torch.Tensor(np.stack(frames))

        # Pull video class
        dataset_id = video_id_to_dataset_id(video_id)
        video_annotations = self.annotations.get(dataset_id, {})
        default_class_id = len(self.class_map)
        video_class = video_annotations.get("class_id", default_class_id)

        # Load audio
        audio = get_audio_features_by_video_id(video_id, self.data_dir)
        return video_frames_tensor, audio, video_class


class VideoDatasetCollator:
    def __call__(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take incoming (video_frames_tensor, audio, video_class) and collates them.
        """
        # Collate video frames
        video_frames = [x[0] for x in batch]
        video_frames = torch.nn.utils.rnn.pad_sequence(video_frames, batch_first=True)

        # Collate audio waveforms
        audio_waves = [x[1] for x in batch]
        audio_waves = torch.nn.utils.rnn.pad_sequence(audio_waves, batch_first=True)

        # Collate labels
        labels = torch.LongTensor([x[2] for x in batch])
        return video_frames, audio_waves, labels


@lru_cache(maxsize=None)
def frame_normalizer(height=240, width=320, grayscale=True):
    # Composes a set of transform functions
    def transform_composer(*fns):
        def inner(frame: np.ndarray):
            for fn in fns:
                frame = fn(frame)
            return frame

        return inner

    # Resizes a frame to a given (height, width)
    def resizer(frame: np.ndarray):
        return cv2.resize(frame, (width, height))

    # Grayscales a frame
    def grayscaler(frame: np.ndarray):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    identity = lambda x: x
    return transform_composer(
        resizer,
        grayscaler if grayscale else identity,
    )


def load_data(
    batch_size=32, vid_height=240, vid_width=360, frame_skip=10, grayscale=True
) -> tuple[DataLoader, DataLoader]:
    train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
    test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
    train_dataset = VideoDataset(
        train_dir,
        transform=frame_normalizer(
            height=vid_height, width=vid_width, grayscale=grayscale
        ),
        frame_skip=frame_skip,
    )
    test_dataset = VideoDataset(
        test_dir,
        transform=frame_normalizer(
            height=vid_height, width=vid_width, grayscale=grayscale
        ),
        frame_skip=frame_skip,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=VideoDatasetCollator(),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=VideoDatasetCollator(),
    )
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    load_annotations_and_classmap()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", default=1000, type=int)
    parser.add_argument("--n_test", default=200, type=int)
    config = parser.parse_args()
    n_train = config.n_train
    n_test = config.n_test
    train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
    test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
    download_data(train_dir, test_dir, n_train, n_test)
