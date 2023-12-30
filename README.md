# Keypress Analysis
This repository is created for the technical task for PhD candidate in Artificial intelligence for monitoring of Parkinson’s
disease. 

### Code Review

1. preprocessor.py
2. dataloader.py
3. train.py
4. test.py
5. postprocessor.py

## 1. preprocessor.py
The script's primary purpose is to prepare video and audio data for further analysis or machine learning model training, with a specific focus on the detection of keypress events. It demonstrates an efficient approach to handling and organizing video and audio data for such specialized tasks.

# Video Data Preprocessing Script for Keypress Detection

This Python script is designed for preprocessing video data for a project focused on keypress detection. It utilizes libraries like `librosa` for audio processing and `moviepy` for video handling.

## Key Functionalities

### Extracting Frames and Audio Frequencies

```python
def extract_frames_and_audio_freq(video_path, output_folder, frame_rate=30):
    ...
```

This function extracts frames and corresponding audio frequencies from a given video file. It saves the frames as images and audio frequencies in a text file, organizing them for further analysis.

### Organizing Extracted Data

```python
def organize_folder(directory):    
    ...
```

The extracted frames and audio data are organized into a structured folder format. This organization aids in the efficient handling of data for subsequent processing or model training.

### Additional Functionalities

- **Extracting Audio from Video**: The script can extract audio tracks from video files for separate analysis.
  
  ```python
  def extract_audio_from_video(video_path, audio_output_path):
      ...
  ```

- **Noise Reduction in Audio**: Although tested, this feature was not used in the main workflow. It involves reducing noise from the extracted audio to potentially improve data quality.

  ```python
  def remove_noise_from_audio(input_audio_path, output_audio_path, noise_start=0, noise_end=10000):
      ...
  ```

- **Plotting Frequency Graphs**: This feature, used for testing, allows for the visualization of audio frequency over time.

  ```python
  def plot_frequency_graph(audio_path):
      ...
  ```

## 2. dataloader.py
This Python script defines a custom dataset class `Custom3DDataset`, which is a subclass of PyTorch's `Dataset` class, tailored for handling a dataset composed of sequences of image frames and associated labels. This dataset is particularly suited for tasks that involve sequences of images, such as video processing or time-series image data. 

- **Custom3DDataset Class**:
  - `__init__`: Initializes the dataset object with the directories of frames and labels, and an optional transform.
  - `_load_samples`: Private method to load and pair frames with their corresponding labels. It organizes the data into a list of tuples, where each tuple contains paths to the image frames and their associated labels.
  - `__len__`: Returns the total number of samples in the dataset.
  - `__getitem__`: Retrieves a sample by its index. It loads the image frames, applies the specified transformations, and stacks them into a tensor. It also converts the label into a tensor.
  - The `transform` variable defines a series of transformations to be applied to each image frame, in this case, just converting images to PyTorch tensors.

This setup is crucial for preparing a dataset for training machine learning models with PyTorch, especially in applications like video analysis or any task that requires handling sequences of images along with their corresponding labels.

## 3. train.py
This Python script implements a deep learning model for processing time-series data, specifically designed for keypress detection in videos. It utilizes a convolutional neural network (CNN) architecture, modifying a pre-trained ResNet-18 model to fit the specific requirements of the task. The model, named `TimeSeriesCNN`, is adapted to handle multiple frames with multiple channels by adjusting the first convolutional layer of ResNet and removing its final fully connected layer. The script also includes temporal layers for processing time-series data and a regression layer for output. Data loading is handled through a custom dataset (`d.Custom3DDataset`), and the training loop involves forward passes, loss calculation using mean squared error, backward passes for gradient calculation, and optimization steps. The model's parameters are optimized using the Adam optimizer. Finally, the model is saved and evaluated, with predictions made on the data. This approach highlights the integration of convolutional and temporal processing for time-series data analysis, particularly in the context of video analysis for keypress detection.

```python
class TimeSeriesCNN(nn.Module):
    # ... (model definition)
    def forward(self, x):
        # ... (forward pass logic)

model = TimeSeriesCNN()
# ...

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(data_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ...
torch.save(model.state_dict())
model.eval()
# ...
```

The code exemplifies how to define, train, and evaluate a deep learning model for a specific application, showcasing the integration of modified pre-trained models with custom data processing.

## 4. test.py

The `calculate_metrics` function in Python is designed for evaluating the performance of predictive models by comparing the predicted values (`y_pred`) with the true values (`y_true`). It computes three essential metrics:

1. **Mean Absolute Error (MAE)**: This metric measures the average magnitude of the errors in a set of predictions, without considering their direction. It's calculated using `mean_absolute_error(y_true, y_pred)` from `sklearn.metrics`.

    ```python
    mae = mean_absolute_error(y_true, y_pred)
    ```

2. **Root Mean Squared Error (RMSE)**: RMSE provides a measure of how well a model's predictions approximate the actual observations. The errors are squared before averaging, which emphasizes larger errors. This is calculated by taking the square root of `mean_squared_error(y_true, y_pred)`.

    ```python
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ```

3. **Dynamic Time Warping (DTW) Distance**: DTW is a more advanced metric used primarily in time series analysis, capturing the similarity between two temporal sequences which may vary in speed. The `fastdtw` method from the `fastdtw` library, coupled with the `euclidean` distance from `scipy.spatial.distance`, is used to compute this distance.

    ```python
    dtw_distance, _ = fastdtw(y_true, y_pred, dist=euclidean)
    ```

These metrics collectively provide a comprehensive assessment of a model's accuracy and predictive capability, encompassing both linear discrepancies (MAE, RMSE) and temporal dynamics (DTW).

## 5. postprocessor.py

This script is designed for processing a video file to detect specific events characterized by significant changes in frequency values within the video frames. It is implemented using the OpenCV library for video processing and analysis.