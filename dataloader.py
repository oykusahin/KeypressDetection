import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Custom3DDataset(Dataset):
    def __init__(self, frames_dir, labels_dir, transform=None):
        self.frames_dir = frames_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for label_file in sorted(os.listdir(self.labels_dir)):
            frame_folder = label_file.split('.')[0]  # Assuming label file format is '1.txt'
            frame_folder_path = os.path.join(self.frames_dir, frame_folder)

            label_file_path = os.path.join(self.labels_dir, label_file)
            with open(label_file_path, 'r') as file:
                labels = [float(line.strip()) for line in file.readlines()]

            frames = [os.path.join(frame_folder_path, f) for f in sorted(os.listdir(frame_folder_path))]
            samples.append((frames, labels))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, labels = self.samples[idx]
        frames = [Image.open(frame_path) for frame_path in frame_paths]
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames_stack = torch.stack(frames)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        return frames_stack, labels_tensor

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other transformations here.
])

# Create the DataLoader
frames_dir = '/Users/oyku/Documents/Projects/KeypressDetection/output/images'
labels_dir = '/Users/oyku/Documents/Projects/KeypressDetection/output/labels'
dataset = Custom3DDataset(frames_dir, labels_dir, transform=transform)
sample_frames, sample_labels = dataset[0]
print("Shape of frames:", sample_frames.shape)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
