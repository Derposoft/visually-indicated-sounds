import cv2
import os
import numpy as np

def load_and_preprocess_data(labels_file, video_directory, num_frames=15):
    frames_list = []  # Store all frames as a list
    labels_list = []  # Store corresponding labels

    with open(labels_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            video_name, label = line.strip().split(',')
            labels_list.append(int(label))
            video_path = os.path.join(video_directory, video_name)

            if '.DS_Store' in video_path:
                continue

            cap = cv2.VideoCapture(video_path)

            frames = []

            for j in range(num_frames):
                frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * j / num_frames)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

            while len(frames) < num_frames:
                frames.append(frames[-1])

            frames_list.append(frames)

    # Convert the lists to NumPy arrays
    frames_array = np.array(frames_list)
    labels_array = np.array(labels_list)

    return frames_array, labels_array


if __name__ == "__main__":

    labels_file = 'train_labels.txt'
    video_directory = '../data/vig_train/'
    num_frames=15

    train_frames_list, train_labels_list = load_and_preprocess_data(labels_file, video_directory, num_frames)

    print(train_frames_list)
    print(train_labels_list)
