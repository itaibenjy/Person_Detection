import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

def process_frame(frame):
    """Process a single frame through YOLOv5."""
    # Convert the BGR frame to RGB and wrap in a list for YOLOv5
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model([frame_rgb], size=640)  # You can adjust the size
    
    # Render detections (this modifies the images in-place)
    rendered_imgs = results.render()
    
    
    # Access the first rendered image
    frame_rgb_with_detections = rendered_imgs[0]

    # Convert RGB back to BGR for OpenCV
    frame_processed = cv2.cvtColor(frame_rgb_with_detections, cv2.COLOR_RGB2BGR)
    return frame_processed


def process_video(input_path, output_path, frame_rate=12):
    """Extract frames, process through YOLOv5, and reassemble into a video."""
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    input_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(input_frame_rate / frame_rate)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (frame_width, frame_height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 output
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, out_size)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_count % frame_skip == 0:
            frame_processed = process_frame(frame)
            out.write(frame_processed)

        frame_count += 1

    # Release everything
    cap.release()
    out.release()

# Example usage
input_video_path = 'sample.mov'
output_video_path = 'output.mp4'
process_video(input_video_path, output_video_path)

