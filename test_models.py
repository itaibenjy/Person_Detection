import torch
from PIL import Image
import matplotlib.pyplot as plt
import time
import sys

# Define the models to test
models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']

# Image to test
image_path = sys.argv[1]  # Update this to your image path
img = Image.open(image_path)
width, height = img.size
title = f'{width}x{height} {image_path.split(".")[0]}'

# Stats storage
detection_times = []
person_counts = []
average_confidences = []

# Prepare a figure to show each model's output image
fig_img, axs_img = plt.subplots(1, len(models), figsize=(40, 10))
fig_img.suptitle(f'Output Images for Different YOLOv5 Models {title}', fontsize=16)

for i,model_name in enumerate(models):
    # Load model
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    # Start timing
    start_time = time.time()
    # Perform inference
    results = model(img)
    # End timing
    end_time = time.time()

    # Filter for persons (class 0)
    persons = results.xyxy[0][results.xyxy[0][:, -1] == 0]  # Filter out all non-person detections

    # Calculate stats
    detection_time = end_time - start_time
    person_count = persons.shape[0]
    average_confidence = persons[:, 4].mean().item() if person_count > 0 else 0

    # Store stats
    detection_times.append(detection_time)
    person_counts.append(person_count)
    average_confidences.append(average_confidence)
    
    results.render()
    img_with_detections = results.ims[0]  # Get the first (and only) image after rendering detections

    # Convert array to Image for display
    img_pil = Image.fromarray(img_with_detections)
    
    # Display the output image
    axs_img[i].imshow(img_pil)
    axs_img[i].axis('off')  # Turn off axis
    axs_img[i].set_title(model_name)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{title} images.jpeg", dpi=300)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Detection time plot
axs[0].bar(models, detection_times, color='skyblue')
axs[0].set_title('Detection Time (s)')
axs[0].set_ylabel('Time (s)')
axs[0].set_xlabel('Model')
axs[0].set_xticklabels(models, rotation=90)

# Person count plot
axs[1].bar(models, person_counts, color='lightgreen')
axs[1].set_title('Number of Persons Detected')
axs[1].set_ylabel('Count')
axs[1].set_xlabel('Model')
axs[1].set_xticklabels(models, rotation=90)

# Average confidence plot
axs[2].bar(models, average_confidences, color='salmon')
axs[2].set_title('Average Confidence in Person Detection')
axs[2].set_ylabel('Confidence')
axs[2].set_xlabel('Model')
axs[2].set_xticklabels(models, rotation=90)

fig.suptitle(title, fontsize=16)
plt.tight_layout()
plt.savefig(f"{title}.jpeg")
