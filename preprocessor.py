from moviepy.editor import VideoFileClip
import cv2
import librosa
import numpy as np
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = f"{output_folder}/frame_{count}.jpg"
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    return frame_rate

def detect_keypresses(audio_path, frame_rate):
    y, sr = librosa.load(audio_path, sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    keypress_frames = np.round(onset_times * frame_rate).astype(int)
    return keypress_frames

def label_frames(video_path, frame_output_folder, audio_output_path):
    if not os.path.exists(frame_output_folder):
        os.makedirs(frame_output_folder)

    video_clip = VideoFileClip(video_path)

    if not audio_output_path.endswith('.wav'):
        audio_output_path += '.wav'

    video_clip.audio.write_audiofile(audio_output_path, codec='pcm_s16le')

    frame_rate = extract_frames(video_path, frame_output_folder)
    keypress_frames = detect_keypresses(audio_output_path, frame_rate)

    frame_labels = {f'frame_{i}.jpg': 1 if i in keypress_frames else 0 for i in range(int(video_clip.duration * frame_rate))}

    return frame_labels


video_path = '/Users/oyku/Documents/Projects/KeypressDetection/data/data1.mp4'
frame_output_folder = '/Users/oyku/Documents/Projects/KeypressDetection/output/frame'
audio_output_path = '/Users/oyku/Documents/Projects/KeypressDetection/output/audio'

frame_labels = label_frames(video_path, frame_output_folder, audio_output_path)
