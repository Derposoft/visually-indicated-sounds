import os, sys
import numpy as np
from tqdm import tqdm
from pytube import YouTube
import argparse

root_url = "https://www.youtube.com/watch?v="
num_all_video = 16384
err_ids = []
completed_ids = []


def download_video(video_id: str, dir: str):
    # URL of the YouTube video you want to download
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(video_url)
    filename = f"{video_id}.mp4"
    video_stream = yt.streams.get_lowest_resolution()
    video_stream.download(output_path=dir, filename=filename)


def download_data(train_dir: str, test_dir: str, num_train: int = 5, num_test: int = 2):
    max_videos = num_train + num_test
    video_id_file = os.path.join(os.path.dirname(__file__), "vig_dl.lst")
    with open(video_id_file) as fin:
        lines = fin.readlines()
        for video_idx, line in tqdm(enumerate(lines[:max_videos])):
            video_id = line.strip().split(",")[0]
            download_dir = train_dir if video_idx < num_train else test_dir
            print(download_dir)
            try:
                download_video(video_id, train_dir, download_dir)
            except:
                print(f"DL Failed, id={video_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", default=5, type=int)
    parser.add_argument("--n_test", default=2, type=int)
    config = parser.parse_args()
    n_train = config.n_train
    n_test = config.n_test
    train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
    test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
    download_data(train_dir, test_dir, n_train, n_test)
