import argparse
import cv2
from functools import lru_cache
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import os
import pickle as pkl
from pytube import YouTube
import scipy.signal as signal
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as audiotransforms
from tqdm import tqdm


train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")


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


def match_seq_len(
    tgt: torch.Tensor, src: torch.Tensor, verbose: bool = False
) -> torch.Tensor:
    """
    Weird hacky function to make src and tgt match seq lens. Original POCAN
    paper just assumes that the clip and generated audio are the same length
    so this is a way to deal with that. Since this is a more general utility
    function that is also used in this file, I'm putting it here.

    :param tgt: target sequence to match shape to, of size (seq_len, dim)
    :param src: source sequence to mold, of size (batch_size, seq_len', dim)
    """
    tgt_seq_len, tgt_dim = tgt.size()
    src_bs, src_seq_len, src_dim = src.size()
    ALLOWABLE_PAD_THRESHOLD = tgt_seq_len * 0.2
    pad_amount = tgt_seq_len - src_seq_len

    # assert tgt_bs == src_bs, "ERROR: bad batch sizes in internal calculations"
    assert tgt_dim == src_dim, "ERROR: bad dim in internal calculations"

    if pad_amount < 0:
        src = src[:, :tgt_seq_len, :]
    elif pad_amount > 0:
        src = F.pad(src, (0, 0, 0, pad_amount))

    # Just in case this function actually doesn't do what we want :^)
    if verbose and abs(pad_amount) > ALLOWABLE_PAD_THRESHOLD:
        print(
            "***WARNING*** utils.py: Exceeding spectrogram synthesizer pad threshold! "
            f"threshold={ALLOWABLE_PAD_THRESHOLD} -- pad={pad_amount}"
        )
    return src


def get_audio_waveform_by_video_id(
    video_id: str, data_dir: str, verbose: bool = False, mono: bool = True
):
    """TODO doesn't support stereo audio."""
    file_path = os.path.join(data_dir, get_filename_from_video_id(video_id))
    if not os.path.exists(file_path):
        if verbose:
            print("get_audio_waveform_by_video_id: called for undownloaded video_id.")
        try:
            _ = download_video(video_id, data_dir)
        except:
            print("Download failure.")
            return None, None
    assert os.path.exists(file_path), ""
    waveform, sample_rate = torchaudio.load(file_path)

    # Enforce mono audio
    if mono:
        waveform = waveform[0]
    return waveform, sample_rate


def get_audio_features_by_video_id(
    video_id: str, data_dir: str, model: str, n_mels: int = 32
) -> torch.Tensor:
    """
    :param video_id: ID of the youtube video
    :param data_dir: Directory that this video is in
    :param model: The type of model (passed in throught the CLI) for any model-specific
        preprocessing (or lack thereof).
    :param n_mels: The number of mel filterbanks to use when generating
        the mel spectrogram. A too-high granularity can cause empty filterbank bins (lots of 0s)
        which can bias models. A too-low granularity can reduce the quality of the input,
        and also negatively affect models.

    Perform audio feature generation as described by "Visually Indicated Sounds", Owens et al.
    TODO different audio feature extraction for each model, if we have time

    Steps:
    1. Apply band-pass filters w/ ERB spacing
    2. Compute subband envelopes
    3. Downsample and compress envelopes
    """

    # Get waveform from video_id, set algorithm parameters
    waveform, sample_rate = get_audio_waveform_by_video_id(video_id, data_dir)
    out_rate = 90
    compression_constant = 0.3

    # Perform special audio feature extraction by model
    if model == "pocan":
        return waveform

    # Apply filter, compute envelope. Here we choose mel filterbank
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels
    )
    spectrogram = mel_spectrogram(waveform)
    spectrogram_tensor = torch.log1p(spectrogram)
    spectrogram = spectrogram_tensor.numpy()
    envelope = np.imag(signal.hilbert(spectrogram, axis=0))
    envelope = torch.Tensor(envelope)

    # Downsample and compress the envelopes
    downsampling = audiotransforms.Resample(orig_freq=sample_rate, new_freq=out_rate)
    downsampled_envelopes = downsampling(envelope)
    compressed_envelopes = torch.abs(downsampled_envelopes) ** compression_constant

    # Output audio as (seq_len, feats) instead of (feats, seq_len)
    compressed_envelopes = compressed_envelopes.permute(1, 0)
    return compressed_envelopes


def video_id_to_dataset_id(video_id: str) -> int:
    map = load_video_ids()
    return int(map[video_id])


def dataset_id_to_video_id(dataset_id: int) -> str:
    map = load_video_ids()
    inverse_map = {}
    for k in map:
        inverse_map[int(map[k])] = k
    return inverse_map[dataset_id]


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


@lru_cache(maxsize=None)
def create_default_spectrograms(
    sample_size=1, model="POCAN", verbose=False, n_fft=400
) -> dict[int, torch.Tensor]:
    """
    Function that generates default spectrograms for each audio class.
    For simplicity, we consider the "default" to simply be the spectrogram of a
    single random sample during testing.

    Returns a dictionary of class ids to complex-valued spectrogram.
    """
    # For default spectrogram caching
    default_spectrograms_filename = f"default_spectrograms_{sample_size}-samples.pkl"
    default_spectrograms_path = os.path.join(
        os.path.dirname(__file__), default_spectrograms_filename
    )
    if os.path.exists(default_spectrograms_path):
        with open(default_spectrograms_path, "rb") as f:
            return pkl.load(f)

    # Find a sample for each class instance and create a spectrogram for it
    default_spectrograms = {}
    annotations, class_map = load_annotations_and_classmap()
    spectrogram_transform = audiotransforms.Spectrogram(n_fft=n_fft, power=None)
    for c in class_map:
        c_id = class_map[c]
        n_samples = 0
        spectrograms = []

        # Collect sample_size number of spectrograms for each class and average them
        for dataset_id in annotations:
            video_annotations = annotations[dataset_id]
            default_class_id = len(class_map)
            video_class = video_annotations.get("class_id", default_class_id)
            if video_class == c_id:
                # Attempt to download this video and use its spectrogram
                video_id = dataset_id_to_video_id(dataset_id)
                waveform, sample_rate = get_audio_waveform_by_video_id(
                    video_id, train_dir
                )
                if waveform == None:
                    continue
                spectrogram: torch.Tensor = spectrogram_transform(waveform)
                spectrograms.append(spectrogram)
                n_samples += 1
            if n_samples >= sample_size:
                break

        # Average collected spectrograms
        if len(spectrograms) == 0:
            raise ValueError(
                f"utils.py: Spectrogram list empty! No videos of class {c} found!"
            )
        elif verbose:
            print(f"Created default spectrogram for {c}")
        longest_spectrogram_len = max(
            spectrogram.shape[-1] for spectrogram in spectrograms
        )
        padded_spectrograms = [
            F.pad(
                spectrogram, (0, longest_spectrogram_len - spectrogram.shape[-1], 0, 0)
            )
            for spectrogram in spectrograms
        ]
        avg_spectrogram = torch.stack(padded_spectrograms).sum(dim=0)
        avg_spectrogram = avg_spectrogram / len(avg_spectrogram)
        default_spectrograms[c_id] = avg_spectrogram.permute(1, 0)  # (seq_len, dim)

    # Cache default spectrograms
    with open(default_spectrograms_path, "wb") as f:
        pkl.dump(default_spectrograms, f)
    return default_spectrograms


class VideoDataset(Dataset):
    def __init__(
        self,
        model: str,
        data_dir: str = os.path.join(os.path.dirname(__file__), "./vig"),
        transform: bool = None,
        frame_skip: int = 10,
    ):
        self.model = model
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
        audio = get_audio_features_by_video_id(video_id, self.data_dir, self.model)
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
    model: str,
    batch_size=32,
    vid_height=240,
    vid_width=360,
    frame_skip=10,
    grayscale=True,
) -> tuple[DataLoader, DataLoader]:
    train_dir = os.path.join(os.path.dirname(__file__), "./vig_train")
    test_dir = os.path.join(os.path.dirname(__file__), "./vig_test")
    train_dataset = VideoDataset(
        model,
        train_dir,
        transform=frame_normalizer(
            height=vid_height, width=vid_width, grayscale=grayscale
        ),
        frame_skip=frame_skip,
    )
    test_dataset = VideoDataset(
        model,
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
    download_data(train_dir, test_dir, n_train, n_test)
