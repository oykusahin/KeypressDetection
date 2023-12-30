import cv2
import numpy as np

def process_frame(frame):
    pass

def add_magenta_dot(frame):
    dot_size = 20
    magenta_color = (255, 0, 255)  

    roi = frame[frame_height - dot_size:, frame_width - dot_size:]

    roi[:, :] = magenta_color

video_file = "data2_nosound.mp4"
cap = cv2.VideoCapture(video_file)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

output_file = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height), isColor=True)

prev_frequency = None
downfall_threshold = 50  
downfall_detected = False

time_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frequency_value = process_frame(frame)  
    if prev_frequency is not None:
        if prev_frequency - frequency_value > downfall_threshold:
            downfall_detected = True
            time_points.append(cap.get(cv2.CAP_PROP_POS_MSEC))  

    prev_frequency = frequency_value

    if downfall_detected:
        frame = add_magenta_dot(frame)

    output_video.write(frame)

cap.release()
output_video.release()

with open("data2_estimated_labels.txt", "w") as file:
    for time_point in time_points:
        file.write(f"{time_point:.2f}\n")

cv2.destroyAllWindows()