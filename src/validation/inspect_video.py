import cv2
import json
from pathlib import Path

# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to subject3 video
VIDEO_PATH = PROJECT_ROOT / "datasets" / "raw" / "subject3" / "vid.avi"

# Open the video using OpenCV
cap = cv2.VideoCapture(str(VIDEO_PATH))

print("\nSUBJECT 3 - VIDEO INSPECTION")
print("-" * 45)

# Check whether OpenCV opened the video
if not cap.isOpened():
    raise RuntimeError("Could not open the video using OpenCV.")

# Extract video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate video duration
duration = frame_count / fps if fps > 0 else 0

print("Video opened successfully.")
print("FPS:", fps)
print("Total frames:", frame_count)
print("Resolution:", width, "x", height)
print("Duration:", round(duration, 3), "seconds")

# Decode five frames to verify that the video is readable
decoded_frames = 0

for i in range(5):
    success, frame = cap.read()

    if success and frame is not None:
        decoded_frames += 1
        print(f"Frame {i + 1}: decoded successfully - Shape: {frame.shape}")
    else:
        print(f"Frame {i + 1}: decoding failed")
        break

# Release video resources
cap.release()

# Save video metadata for later pipeline stages
metadata = {
    "subject": "subject3",
    "video_path": str(VIDEO_PATH),
    "fps": fps,
    "frame_count": frame_count,
    "width": width,
    "height": height,
    "duration_seconds": duration,
    "decoded_test_frames": decoded_frames
}

OUTPUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "subject3_video_metadata.json"

with open(OUTPUT_PATH, "w") as file:
    json.dump(metadata, file, indent=4)

print("\nMetadata saved to:", OUTPUT_PATH)
print("Video inspection completed.")