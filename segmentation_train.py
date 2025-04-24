# Copyright (c) 2025 Mark A. Bowers
# All rights reserved.
#
# IGVC Perception Project
# https://github.com/mabowers/igvc-perception
# ECE5532 Autonomous Vehicle Systems I, Winter 2025
# Oakland University, Rochester, MI
# Instructor: Dr. Micho Radovnikovich
#
# This script trains the YOLO11 segmentation network on the IGVC dataset
#
# References used for this file:
# https://docs.ultralytics.com/tasks/segment/

from ultralytics import YOLO

# Use the YOLO11 small model (10.1M params, 25.5 GFLOPs)
model = YOLO("yolo11s-seg.yaml").load("yolo11s.pt")  # build from YAML and transfer pre-trained weights

# Train the model
results = model.train(data="igvc-annotated/data.yaml", epochs=100, imgsz=640, batch=16)
