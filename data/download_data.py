import os, sys
import numpy as np
from tqdm import tqdm
from pytube import YouTube
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--num_videos", default=10, type=int)
config = parser.parse_args()
max_videos = config.num_videos


root_url = "https://www.youtube.com/watch?v="
num_all_video = 16384
err_ids = []
completed_ids = []
download_folder = os.path.join(__file__, "./")


def download_video(video_id):
    # URL of the YouTube video you want to download
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(video_url)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(output_path=download_folder)


with open("vig_dl.lst") as fin:
    lines = fin.readlines()
    for line_id, line in tqdm(enumerate(lines[:max_videos])):
        print("Downloading")
        video_id = line.strip().split(",")[0]
        try:
            download_video(video_id)
            completed_ids.append(video_id)
        except:
            print("%d/%d: %s fail downloading" % (line_id, num_all_video, video_id))
            err_ids.append(video_id)

err_ids = np.array(err_ids)
completed_ids = np.array(completed_ids)
