import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image
import dataloader as d
import os

class TimeSeriesCNN(nn.Module):
    def __init__(self, num_frames=10, num_channels=4, pretrained=True):
        super(TimeSeriesCNN, self).__init__()
        
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove the final FC layer

        self.temporal_layers = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=3, stride=1, padding=1),  # Changed input channels to 512
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.regression = nn.Linear(256 * (num_frames // 4), 1)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        x = x.view(batch_size * num_frames, C, H, W)  # Reshape for processing by ResNet

        features = self.resnet(x)  # Shape: [batch_size * num_frames, resnet_out_features]
        features = features.view(batch_size, num_frames, -1)  # Reshape to [batch, frames, features]
        features = features.permute(0, 2, 1)  # Reshape to [batch, features, frames] for Conv1d
        temporal_features = self.temporal_layers(features)

        output = temporal_features.view(batch_size, -1)
        output = self.regression(output)

        return output.squeeze() 

model = TimeSeriesCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the DataLoader
mainDir = '/Users/oyku/Documents/Projects/KeypressDetection/'
framesDir = os.path.join(mainDir, 'output/images')
labelsDir = os.path.join(mainDir, 'output/labels')
dataset = d.Custom3DDataset(framesDir, labelsDir, transform=d.transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
sample_frames, sample_labels = dataset[0]
print("Shape of frames:", sample_frames.shape)


num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(data_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict())
model.eval()
predictions = []
with torch.no_grad():
    for inputs in data_loader:
        outputs = model(inputs)
        predictions.append(outputs.round()) 
