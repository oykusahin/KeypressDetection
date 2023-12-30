import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os

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

class TimeSeriesCNN(nn.Module):
    def __init__(self, num_frames=10, num_channels=4, pretrained=True):
        super(TimeSeriesCNN, self).__init__()
        # Load a pretrained ResNet and replace the top layer for feature extraction
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove the final FC layer

        # Adjust the input channels of the first Conv1d layer to match ResNet's output features
        self.temporal_layers = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=3, stride=1, padding=1),  # Changed input channels to 512
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Final regression layer
        self.regression = nn.Linear(256 * (num_frames // 4), 1)

    def forward(self, x):
        # x shape: [batch, frames, channels, height, width]
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(batch_size * num_frames, C, H, W)  # Reshape for processing by ResNet

        # Feature extraction for each frame
        features = self.resnet(x)  # Shape: [batch_size * num_frames, resnet_out_features]
        features = features.view(batch_size, num_frames, -1)  # Reshape to [batch, frames, features]

        # Temporal processing
        features = features.permute(0, 2, 1)  # Reshape to [batch, features, frames] for Conv1d
        temporal_features = self.temporal_layers(features)

        # Flatten and pass through the regression layer
        output = temporal_features.view(batch_size, -1)
        output = self.regression(output)

        return output


        
# Initialize the model, loss function, and optimizer
model = TimeSeriesCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example DataLoader (replace with your data loader)
#train_loader = DataLoader(datasets.FakeData(transform=transforms.ToTensor()), batch_size=8, shuffle=True)



# Create the DataLoader
frames_dir = '/Users/oyku/Documents/Projects/KeypressDetection/output/images'
labels_dir = '/Users/oyku/Documents/Projects/KeypressDetection/output/labels'
dataset = Custom3DDataset(frames_dir, labels_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

num_epochs = 1
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(data_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

