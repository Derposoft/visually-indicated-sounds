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

root_url = "https://www.youtube.com/watch?v="


def download_video(video_id: str, dirname: str):
    # URL of the YouTube video you want to download
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(video_url)
    filename = f"{video_id}.mp4"
    video_stream = yt.streams.get_lowest_resolution()
    video_stream.download(output_path=dirname, filename=filename)
    return filename

def download_data(train_dir: str, test_dir: str, num_train: int = 5, num_test: int = 2):
    max_videos = num_train + num_test
    video_id_file = os.path.join(os.path.dirname(__file__), "vig_dl.lst")

    annotations_dict, classmap_dict = load_annotations_and_classmap()

    with open(video_id_file) as fin:
        lines = fin.readlines()
        for video_idx, line in tqdm(enumerate(lines[:max_videos])):
            youtube_video_id, video_id = line.strip().split(",")
            video_id = int(video_id)

            download_dir = train_dir if video_idx < num_train else test_dir

            try:
                download_video_filename = download_video(youtube_video_id, download_dir)
            except:
                continue

            download_video_saved_path = os.path.join(download_dir, download_video_filename)
            clipped_video_saved_path = os.path.join(download_dir, "clipped_" + download_video_filename)

            print()

            if video_id in annotations_dict:
                desired_annotation = annotations_dict[video_id]

                start_time = int(desired_annotation['start_time'])
                end_time = int(desired_annotation['end_time'])

                print("start_time:", start_time)
                print("end_time:", end_time)

                ffmpeg_extract_subclip(download_video_saved_path, start_time, end_time, targetname=clipped_video_saved_path)

                os.remove(download_video_saved_path)
                shutil.move(clipped_video_saved_path, download_video_saved_path)


def download_data_if_not_downloaded(
    data_dir=os.path.join(os.path.dirname(__file__), "./vig_train"),
    n_train_videos=1000,
    n_test_videos=200,
):
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
    if not data_files:
        train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
        test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
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


def video_id_to_dataset_id(video_id):
    map = load_video_ids()
    return map[video_id]


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
        self, data_dir=os.path.join(os.path.dirname(__file__), "./vig"), transform=None
    ):
        self.data_dir = data_dir
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
        self.transform = transform
        (
            self.annotations,
            self.class_map,
        ) = load_annotations_and_classmap()

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_id = self.video_files[idx].split(".")[0]
        video_file = os.path.join(self.data_dir, self.video_files[idx])
        cap = cv2.VideoCapture(video_file)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        return (
            torch.stack(torch.Tensor(frames)),
            self.annotations[video_id],
            self.class_map[self.annotations[video_id]["vig_label"]],
        )


def load_data(batch_size=32) -> tuple[DataLoader, DataLoader]:
    train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
    test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
    train_data = DataLoader(VideoDataset(train_dir), batch_size=batch_size)
    test_data = DataLoader(VideoDataset(test_dir), batch_size=batch_size)
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
