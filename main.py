import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image
image_path = 'images.jpeg'  # Make sure to replace this with the actual path to your image
img = Image.open(image_path)

# Inference
results = model(img)

# Results
results.print()  # Print results to console
'''

# Extract data for persons (class 0 in COCO)
persons = results.xyxy[0]  # img1 predictions (tensor)
persons = persons[persons[:, 5] == 0]  # Filter for persons

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
ax.imshow(img)

# Iterate over each detected person
for i in range(persons.shape[0]):
    bbox = persons[i].cpu()  # Move the tensor to CPU
    xmin, ymin, xmax, ymax = bbox[:4]  # Extract coordinates
    rect = patches.Rectangle((xmin.item(), ymin.item()), (xmax - xmin).item(), (ymax - ymin).item(), linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.savefig("output.jpeg")

'''
# Get predictions tensor
preds = results.pred[0]

# Filter out non-person detections (class 0 for COCO is person)
# Note: This sets all other detections to zero, effectively removing them
person_preds = preds[(preds[:, 5] == 0) | (preds[:, 4] == 0)]  # Keep scores for persons or zero-score entries

# Manually update the preds tensor in the results
results.pred = [person_preds]

# Render the modified results with only person detections
img_with_detections = results.render()[0]

# Convert the rendered image back to a PIL Image and save it
output_img = Image.fromarray(img_with_detections)
output_img.save('output.jpeg')
