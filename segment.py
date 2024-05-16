import os
import torch
import cv2
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import base64

HOME = os.getcwd()
print("HOME:", HOME)
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

IMAGE_NAME = "dog.jpeg"
IMAGE_PATH = os.path.join(HOME, "data", IMAGE_NAME)

# Load the image
image_rgb = cv2.imread(IMAGE_PATH)

# Define the bounding box
default_box = {'x': 133, 'y': 222, 'width': 562, 'height': 626, 'label': ''}

# Set the box
box = default_box
box = np.array([
    box['x'],
    box['y'],
    box['x'] + box['width'],
    box['y'] + box['height']
])
box = np.array([39, 242, 620, 900])
# Set up the mask predictor and predict masks
mask_predictor.set_image(image_rgb)
masks, scores, logits = mask_predictor.predict(box=box, multimask_output=True)

# Get the largest detected mask
detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)
detections = detections[detections.area == np.max(detections.area)]

# Create a blank canvas with the same size as the original image
foreground_only = np.zeros_like(image_rgb)

# Overlay segmented parts onto the blank canvas
for mask in masks:
    foreground_only[mask > 0] = image_rgb[mask > 0]

# Display the segmented parts
cv2.imshow('Segmented Parts Only', foreground_only)
cv2.waitKey(0)
cv2.destroyAllWindows()
