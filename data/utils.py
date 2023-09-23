import pickle as pkl
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache


from data.download_data import download_data


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
        ) = load_annotations_and_classmap()  # TODO use classmap

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
    train_data = DataLoader(VideoDataset(), batch_size=batch_size)
    test_data = DataLoader(VideoDataset(), batch_size=batch_size)
    return train_data, test_data


def download_data_if_not_downloaded(
    data_dir=os.path.join(os.path.dirname(__file__), "./vig"),
    n_train_videos=10,
    n_test_videos=2,
):
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".mp4")]
    if not data_files:
        train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
        test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
        download_data(train_dir, test_dir, n_train_videos, n_test_videos)


if __name__ == "__main__":
    load_annotations_and_classmap()
