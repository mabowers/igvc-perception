# IGVC perception project
# Mark Bowers
# ECE5532 Autonomous Vehicle Systems I - Winter 2025
# Oakland University
#
# This script reads the IGVC dataset images from img_resized and runs the YOLO11 segmentation network
# to find objects in the scene (i.e. painted lines, generic barrels, traffic barrels, and grass.)
#
# Input images read from:    img_resized/
# Line-only masks saved to:  img_lines/
# Segmented images saved to: img_seg/
# Performance data saved to: performance_data.csv
# Output video saved to:     segmentation_results.mp4
#
# References used for this file:
# https://docs.ultralytics.com/tasks/segment/
# https://github.com/ultralytics/ultralytics
# GitHub Copilot used for general Python and OpenCV reference

import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
import csv
from collections import Counter

# Load model with IGVC-trained weights
model = YOLO("runs/segment/train29/weights/best.pt")

# Directory containing input images
image_dir = "img_resized"

# List of colors for each detection class
colors = [
    (255, 105, 180), # Line    = Pink
    (0, 0, 255),     # Barrel  = Blue
    (255, 165, 0),   # Traffic = Orange
    (0, 255, 0)      # Grass   = Green
]

# List of minimum confidence thresholds for each detection class
class_conf_thresholds = {
    0: 0.65,   # Line
    1: 0.5,    # Barrel
    2: 0.5,    # Traffic
    3: 0.4     # Grass
}

# Define minimum contiguous area thresholds for some detection classes (i.e. lines)
class_min_area_thresholds = {
    0: 1000,  # Line
}

# Get all PNGs in the directory and sort them alphabetically
image_names = sorted([img for img in os.listdir(image_dir) if img.lower().endswith(('.png'))])

# Setup video writer to save a side-by-side output video (original video + segmentation output)
output_video_path = "segmentation_results.mp4"
frame_width = 640+640  # Width of the video frame
frame_height = 640     # Height of the video frame
fps = 10               # Frames per second
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Create output folder for the detected line masks only
line_class_output_folder = "img_lines"
os.makedirs(line_class_output_folder, exist_ok=True)

# Create output folder for the segmented image (all classes)
seg_output_folder = "img_seg"
os.makedirs(seg_output_folder, exist_ok=True)

# List to store performance data for each image
performance_data = []

# Iterate through all sorted images in the directory
for idx, image_name in enumerate(image_names):

    # Get full image path
    image_path = os.path.join(image_dir, image_name)

    # Run inference
    print(f"\nRunning segmentation on {image_name}")
    results = model(image_path)  # Run semantic segmentation on an image (default threshold is 25% confidence)
    result = results[0]          # Get the first and only result (one image is processed at a time)

    # Get neural network timing information
    speeds = result.speed
    preprocess_time_ms  = speeds["preprocess"]
    inference_time_ms   = speeds["inference"]
    postprocess_time_ms = speeds["postprocess"]

    # Load the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    image = original_image.copy()  # Create a copy for overlaying masks of all classes

    # Create a new mask to store all lines that will be detected in this frame
    line_mask = np.zeros_like(image, dtype=np.uint8)

    # Time for filtering this frame
    filtering_time_ms = 0

    # Process and filter detection results
    masks = result.masks             # Get segmentation masks
    classes = result.boxes.cls       # Get class IDs for each mask
    confidences = result.boxes.conf  # Get confidence scores for each mask
    if masks is not None:

        # Iterate through each mask, class ID, and confidence
        for mask, class_id, confidence in zip(masks.data, classes, confidences):

            # Start measuring CPU time for filtering this mask
            filtering_time_mask_start = time.time()

            # convert result types
            #mask = mask.cpu().numpy().astype(np.uint8) # Original implementation
            mask = mask.byte().cpu().numpy()            # Faster implementation (convert to byte on GPU)
            class_id = int(class_id)                    # Convert class ID to int
            confidence = float(confidence)              # Convert confidence to float

            # Check if the confidence meets the threshold required for this class
            if class_id in class_conf_thresholds and confidence < class_conf_thresholds[class_id]:
                continue  # Skip this detection if it doesn't meet the threshold

            # For classes with min area thresholds (i.e. lines)
            if (class_id in class_min_area_thresholds):

                # Create a new mask with only valid components
                filtered_mask = np.zeros_like(mask, dtype=np.uint8)

                # Perform connected component analysis
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

                # Filter out boogers
                for i in range(1, num_labels):                             # Skip the background (label 0)
                    area = stats[i, cv2.CC_STAT_AREA]                      # Get the area of the connected component
                    if area >= class_min_area_thresholds.get(class_id, 0): # Check if the area meets the threshold
                        filtered_mask[labels == i] = 1                     # Add the valid component to the filtered mask

                # If the filtered mask is empty, skip this detection
                if np.sum(filtered_mask) == 0:
                    continue

                # Update the mask to the filtered version
                mask = filtered_mask

            # End measuring CPU time for filtering this mask
            filtering_time_mask_end = time.time()
            filtering_time_mask_ms = (filtering_time_mask_end - filtering_time_mask_start) * 1000
            filtering_time_ms += filtering_time_mask_ms # accumulate time

            if class_id == 0:
                # Add line to the mask
                line_mask[mask == 1] = 255
            else:
                # Remove portions of the line mask that overlap with other detections
                line_mask[mask == 1] = 0

            # Apply the solid mask color directly to the image
            color = colors[class_id % len(colors)]  # Assign a color based on the class ID
            image[mask == 1] = color                # Apply the color to the mask area

    # Print time spent on filtering this frame
    print(f"{filtering_time_ms:.1f}ms filtering")

    # Get the number of detections for each class
    class_ids = [int(class_id) for class_id in classes]
    class_counts = Counter(class_ids)

    # Save performance data
    performance_data.append([
        idx, preprocess_time_ms, inference_time_ms, postprocess_time_ms, filtering_time_ms,
        class_counts.get(0,0), class_counts.get(1, 0), class_counts.get(2, 0), class_counts.get(3, 0)
    ])

    # Save the line mask
    line_mask_uint8 = (line_mask).astype(np.uint8)
    output_path = os.path.join(line_class_output_folder, f"line_mask_{idx:04d}.png")
    cv2.imwrite(output_path, line_mask_uint8)

    # Save the image with masks overlaid
    seg_output_path = os.path.join(seg_output_folder, f"seg_{idx:04d}.png")
    cv2.imwrite(seg_output_path, image)

    # Resize images to fit the video frame dimensions
    original_resized = cv2.resize(original_image, (frame_width // 2, frame_height))
    segmented_resized = cv2.resize(image, (frame_width // 2, frame_height))

    # Concatenate the original and segmented images side by side
    side_by_side = np.concatenate((original_resized, segmented_resized), axis=1)

    # Convert RGB to BGR for OpenCV video writer
    side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)

    # Write the frame to the video
    video_writer.write(side_by_side_bgr)

# Release the video writer
video_writer.release()
print(f"\nVideo saved to {output_video_path}")

# Create CSV file for performance data
logfile_csv = "performance_data.csv"
with open(logfile_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow([
        "Image Index", "Preprocess Time (ms)", "Inference Time (ms)", "Postprocess Time (ms)", "Filtering Time (ms)",
        "Line Count", "Barrel Count", "Traffic Count", "Grass Count"
    ])
    # Write all rows
    writer.writerows(performance_data)
print(f"Performance data saved to {logfile_csv}.")

print("Done :)")
