from moviepy.editor import VideoFileClip
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os


def crop_video(input_video_path, output_video_path, end_time):
    # Load the video
    video = VideoFileClip(input_video_path)
    
    # Crop the first 10 seconds
    cropped_video = video.subclip(0, end_time)
    
    # Write the result to a new file
    cropped_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

# Example usage
input_video_path = '/Users/oyku/Documents/Projects/KeypressDetection/data1.mp4'
output_video_path = '/Users/oyku/Documents/Projects/KeypressDetection/test.mp4'
#crop_video(input_video_path, output_video_path, 10)  # Crop first 10 seconds

def extract_audio_from_video(video_path, audio_output_path):
    # Load the video file
    video = VideoFileClip(video_path)
    
    # Extract the audio
    audio = video.audio
    
    # Write the audio to a file
    audio.write_audiofile(audio_output_path)

# Example usage
video_path = '/Users/oyku/Documents/Projects/KeypressDetection/test.mp4'
audio_output_path = '/Users/oyku/Documents/Projects/KeypressDetection/test_audio.wav'
#extract_audio_from_video(video_path, audio_output_path)


def remove_noise_from_audio(input_audio_path, output_audio_path, noise_start=0, noise_end=10000):
    # Load the audio file
    y, sr = librosa.load(input_audio_path, sr=None)

    # Select a noise segment
    noise_clip = y[noise_start:noise_end]

    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=y, sr=sr)

    # Write the cleaned audio to a new file
    sf.write(output_audio_path, reduced_noise, sr)

# Example usage
input_audio_path = '/Users/oyku/Documents/Projects/KeypressDetection/test_audio.wav'
output_audio_path = '/Users/oyku/Documents/Projects/KeypressDetection/cleaned_audio.wav'
#remove_noise_from_audio(input_audio_path, output_audio_path)

def plot_frequency_graph(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Compute the spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Time variable
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames, sr=sr)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, spectral_centroids, color='b')  # Plot spectral centroid
    plt.title('Spectral Centroid over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

# Example usage
audio_path = '/Users/oyku/Documents/Projects/KeypressDetection/test_audio.wav'
#plot_frequency_graph(audio_path)

def extract_frames_and_audio_freq(video_path, output_folder, frame_rate=30):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load video and audio
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = os.path.join(output_folder, 'temp_audio.wav')
    audio.write_audiofile(audio_path)

    # Load audio for frequency analysis
    y, sr = librosa.load(audio_path, sr=None)

    # Process each frame
    for frame_number in range(int(video.duration * frame_rate)):
        # Extract frame
        frame = video.get_frame(frame_number / frame_rate)
        frame_filename = f"{output_folder}/frame_{frame_number:05d}.png"
        plt.imsave(frame_filename, frame)

        # Extract and analyze audio segment for this frame
        start_sample = int(frame_number / frame_rate * sr)
        end_sample = int((frame_number + 1) / frame_rate * sr)
        y_frame = y[start_sample:end_sample]

        # Calculate mean frequency (spectral centroid) for this frame
        spectral_centroid = librosa.feature.spectral_centroid(y=y_frame, sr=sr)
        mean_frequency = np.mean(spectral_centroid)

        # Save the frequency value
        with open(f"{output_folder}/frame_{frame_number:05d}_freq.txt", 'w') as f:
            f.write(f"{mean_frequency}\n")

    # Remove temporary audio file
    os.remove(audio_path)

# Example usage
video_path = '/Users/oyku/Documents/Projects/KeypressDetection/test.mp4'
output_folder = '/Users/oyku/Documents/Projects/KeypressDetection/output'
#extract_frames_and_audio_freq(video_path, output_folder)

def organize_folder(directory):    
    count = 0
    count2 = 0
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):            
            fileDirectory = os.path.join(directory, filename)
            '''
            if os.path.isfile(fileDirectory):
                textfile = open(fileDirectory, "r")
                audioFrequency = textfile.readline()
                frameNumber = filename.split('_')[1]
                newfilename = int(int(frameNumber) / 10)
                if count >= 10:
                    count = 0
                
                if count == 0:
                    count = count + 1
                    f = open(str(newfilename) + ".txt", "w")
                    f.write(audioFrequency)
                    f.close()

                elif count < 10 and count > 0:
                    count = count + 1
                    f = open(str(newfilename) + ".txt", "a+")
                    f.write(audioFrequency)
                    f.close()
        '''
        if filename.endswith('.png'):
            fileDirectory = os.path.join(directory, filename)
            if os.path.isfile(fileDirectory):
                frameNumber = filename.split('_')[1].rstrip('.png')
                newfilename2 = str(int(int(frameNumber) / 10))

            if count2 >= 10:
                count2 = 0
                
            if count2 == 0:
                count2 = count2 + 1
                path = os.path.join(directory, newfilename2)
                newPath = os.path.join(path, filename) 
                os.mkdir(path)
                os.rename(fileDirectory, newPath)

            elif count2 < 10 and count2 > 0:
                count2 = count2 + 1
                path = os.path.join(directory, newfilename2)
                newPath = os.path.join(path, filename) 
                os.rename(fileDirectory, newPath)
                    

    
    
                

directory = '/Users/oyku/Documents/Projects/KeypressDetection/output'
organize_folder(directory)