import cv2
import numpy as np

# Define the process_frame function to extract frequency values
def process_frame(frame):
    # Implement your logic to process the frame and extract frequency values here
    # Return the extracted frequency value
    pass

# Define the add_magenta_dot function to add a magenta dot to the frame
def add_magenta_dot(frame):
    dot_size = 20
    magenta_color = (255, 0, 255)  # BGR color code for magenta

    # Create a region of interest (ROI) for the dot
    roi = frame[frame_height - dot_size:, frame_width - dot_size:]

    # Fill the ROI with the magenta color
    roi[:, :] = magenta_color

# Load your video file
video_file = "data2_nosound.mp4"
cap = cv2.VideoCapture(video_file)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Create an output video file without sound
output_file = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height), isColor=True)

# Initialize variables for detecting downfall points
prev_frequency = None
downfall_threshold = 50  # Adjust this threshold as needed
downfall_detected = False

# Initialize a list to store time points
time_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame to extract frequency values
    frequency_value = process_frame(frame)  # You need to define this function

    if prev_frequency is not None:
        if prev_frequency - frequency_value > downfall_threshold:
            # A downfall point is detected
            downfall_detected = True
            time_points.append(cap.get(cv2.CAP_PROP_POS_MSEC))  # Record time in milliseconds

    prev_frequency = frequency_value

    # Add a magenta-colored dot if a downfall point is detected
    if downfall_detected:
        frame = add_magenta_dot(frame)

    # Write the frame to the output video
    output_video.write(frame)

cap.release()
output_video.release()

# Save the time points to a text file
with open("data2_estimated_labels.txt", "w") as file:
    for time_point in time_points:
        file.write(f"{time_point:.2f}\n")

# Cleanup
cv2.destroyAllWindows()