import pickle as pkl
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache

import os, sys
import numpy as np
from tqdm import tqdm
from pytube import YouTube
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import shutil


def download_video(video_id: str, dirname: str):
    # URL of the YouTube video you want to download
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(video_url)
    filename = f"{video_id}.mp4"
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


class PadSequence:
    def __call__(self, batch):
        # Sort batch by decreasing length
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        # Pad sequences to max seq len
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        # Get lengths and labels
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.LongTensor([x[1] for x in sorted_batch])
        return sequences_padded, lengths, labels


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
        # print(f"{frame_number} frames; after skipping, {len(frames)} frames")

        # Apply frame size normalization transformation
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        video_frames_tensor = torch.Tensor(np.stack(frames))

        # Pull video class
        dataset_id = video_id_to_dataset_id(video_id)
        video_annotations = self.annotations.get(dataset_id, {})
        default_class_id = len(self.class_map)
        video_class = video_annotations.get("class_id", default_class_id)
        return video_frames_tensor, video_class


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
    batch_size=32, vid_height=240, vid_width=360, grayscale=True
) -> tuple[DataLoader, DataLoader]:
    train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
    test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
    train_data = DataLoader(
        VideoDataset(
            train_dir,
            transform=frame_normalizer(
                height=vid_height, width=vid_width, grayscale=grayscale
            ),
        ),
        batch_size=batch_size,
        collate_fn=PadSequence(),
    )
    test_data = DataLoader(
        VideoDataset(
            test_dir,
            transform=frame_normalizer(
                height=vid_height, width=vid_width, grayscale=grayscale
            ),
        ),
        batch_size=batch_size,
        collate_fn=PadSequence(),
    )
    return train_data, test_data


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
