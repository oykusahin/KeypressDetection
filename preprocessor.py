import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import wavfile
from moviepy.editor import VideoFileClip



def crop_video(input_video_path, output_video_path, end_time):
    video = VideoFileClip(input_video_path)
    cropped_video = video.subclip(0, end_time)
    cropped_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

def extract_audio_from_video(video_path, audio_output_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path)


def remove_noise_from_audio(input_audio_path, output_audio_path, noise_start=0, noise_end=10000):
    y, sr = librosa.load(input_audio_path, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_audio_path, reduced_noise, sr)


def plot_frequency_graph(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames, sr=sr)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, spectral_centroids, color='b')  # Plot spectral centroid
    plt.title('Spectral Centroid over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

def extract_frames_and_audio_freq(video_path, output_folder, frame_rate=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = os.path.join(output_folder, 'temp_audio.wav')
    audio.write_audiofile(audio_path)

    y, sr = librosa.load(audio_path, sr=None)

    for frame_number in range(int(video.duration * frame_rate)):
        frame = video.get_frame(frame_number / frame_rate)
        frame_filename = f"{output_folder}/frame_{frame_number:05d}.png"
        plt.imsave(frame_filename, frame)

        start_sample = int(frame_number / frame_rate * sr)
        end_sample = int((frame_number + 1) / frame_rate * sr)
        y_frame = y[start_sample:end_sample]

        spectral_centroid = librosa.feature.spectral_centroid(y=y_frame, sr=sr)
        mean_frequency = np.mean(spectral_centroid)

        with open(f"{output_folder}/frame_{frame_number:05d}_freq.txt", 'w') as f:
            f.write(f"{mean_frequency}\n")

    os.remove(audio_path)

def organize_folder(directory):    
    countText = 0
    countImage = 0
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):            
            fileDirectory = os.path.join(directory, filename)
            
            if os.path.isfile(fileDirectory):
                textfile = open(fileDirectory, "r")
                audioFrequency = textfile.readline()
                frameNumber = filename.split('_')[1]
                textfilename = int(int(frameNumber) / 10)
                if countText >= 10:
                    countText = 0
                
                if countText == 0:
                    countText = countText + 1
                    f = open(str(textfilename) + ".txt", "w")
                    f.write(audioFrequency)
                    f.close()

                elif countText < 10 and countText > 0:
                    countText = countText + 1
                    f = open(str(textfilename) + ".txt", "a+")
                    f.write(audioFrequency)
                    f.close()
        
        if filename.endswith('.png'):
            fileDirectory = os.path.join(directory, filename)
            if os.path.isfile(fileDirectory):
                frameNumber = filename.split('_')[1].rstrip('.png')
                imagefilename = str(int(int(frameNumber) / 10))

            if countImage >= 10:
                countImage = 0
                
            if countImage == 0:
                countImage = countImage + 1
                path = os.path.join(directory, imagefilename)
                newPath = os.path.join(path, filename) 
                os.mkdir(path)
                os.rename(fileDirectory, newPath)

            elif countImage < 10 and countImage > 0:
                countImage = countImage + 1
                path = os.path.join(directory, imagefilename)
                newPath = os.path.join(path, filename) 
                os.rename(fileDirectory, newPath)

'''
This is the main function of the preprocessor.py file.
It prepares the given data for training by following these steps:
1. Extract audio from video:
2. Extract frames and audio freq
3. Organizes the folder structure
'''                   
def main():
    mainDir = '/Users/oyku/Documents/Projects/KeypressDetection/'
    outputDir = os.path.join(mainDir, 'output')
    videoDir = os.path.join(mainDir, 'rawdata/data1.mp4')
    audioDir = os.path.join(mainDir, 'output/test_audio.wav')
    
    extract_frames_and_audio_freq(videoDir, outputDir)
    organize_folder(outputDir)

    '''
    Tested the performance of noise reducted audio. Did not observed any improvement.
    extract_audio_from_video(videoDir, outputDir)
    remove_noise_from_audio(input_audio_path, output_audio_path)
    plot_frequency_graph(audio_path)
    '''

if __name__=="__main__": 
    main() 