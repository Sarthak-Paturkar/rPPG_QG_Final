import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Extract YuNet facial ROI RGB signals for one subject."
)
parser.add_argument("--subject", default="subject3")
args = parser.parse_args()

if not args.subject.startswith("subject") or not args.subject[7:].isdigit():
    parser.error("--subject must be a subject folder name, such as subject3")

SUBJECT = args.subject

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "datasets" / "raw" / SUBJECT / "vid.avi"

MODEL_PATH = (
    ROOT / "models/yunet/face_detection_yunet_2023mar.onnx"
)

OUTPUT_DIR = ROOT / "datasets" / "processed" / SUBJECT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "rgb_signals.csv"

# --------------------------------------------------
# EXTRACTION SETTINGS
# --------------------------------------------------

SKIN_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)

MIN_SKIN_PIXELS = 30
MIN_SKIN_RATIO = 0.20

# --------------------------------------------------
# INITIALISE YUNET
# --------------------------------------------------

detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH),
    "",
    (640, 480),
    0.8,
    0.3,
    5000
)

# --------------------------------------------------
# OPEN VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Unable to open subject3 video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector.setInputSize((width, height))

print(f"\n{SUBJECT.upper()} - RGB EXTRACTION")
print("-" * 50)
print("FPS:", fps)
print("Total frames:", total_frames)
print("Resolution:", width, "x", height)

# --------------------------------------------------
# ROI CREATION FUNCTION
# --------------------------------------------------

def create_roi(x1, y1, x2, y2, face_box):

    fx, fy, fw, fh = face_box

    # Keep ROI coordinates inside the detected face
    x1 = max(fx, int(round(x1)))
    y1 = max(fy, int(round(y1)))
    x2 = min(fx + fw, int(round(x2)))
    y2 = min(fy + fh, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


# --------------------------------------------------
# CALCULATE THREE FACIAL ROIS
# --------------------------------------------------

def calculate_rois(face):

    x, y, w, h = face[:4].astype(int)

    landmarks = face[4:14].reshape(5, 2)

    right_eye, left_eye, nose, right_mouth, left_mouth = landmarks

    eye_mid = (right_eye + left_eye) / 2
    mouth_mid = (right_mouth + left_mouth) / 2

    eye_distance = np.linalg.norm(right_eye - left_eye)

    if eye_distance < 10:
        return None

    face_box = (x, y, w, h)

    # Forehead: above eyes, avoiding eyebrows and hairline
    forehead = create_roi(
        eye_mid[0] - 0.55 * eye_distance,
        eye_mid[1] - 0.95 * eye_distance,
        eye_mid[0] + 0.55 * eye_distance,
        eye_mid[1] - 0.20 * eye_distance,
        face_box
    )

    # Cheeks: below eyes and above the mouth
    cheek_top = eye_mid[1] + 0.40 * eye_distance
    cheek_bottom = mouth_mid[1] - 0.15 * eye_distance

    cheek_width = 0.55 * eye_distance

    # Subject's right cheek appears on the left of the image
    right_cheek = create_roi(
        right_eye[0] - cheek_width,
        cheek_top,
        right_eye[0] + 0.05 * eye_distance,
        cheek_bottom,
        face_box
    )

    # Subject's left cheek appears on the right of the image
    left_cheek = create_roi(
        left_eye[0] - 0.05 * eye_distance,
        cheek_top,
        left_eye[0] + cheek_width,
        cheek_bottom,
        face_box
    )

    return {
        "forehead": forehead,
        "right_cheek": right_cheek,
        "left_cheek": left_cheek
    }


# --------------------------------------------------
# EXTRACT MEAN RGB FROM ONE ROI
# --------------------------------------------------

def extract_roi_rgb(frame, skin_mask, roi):

    if roi is None:
        return np.nan, np.nan, np.nan, 0.0

    x1, y1, x2, y2 = roi

    # Extract original image pixels and corresponding skin mask
    roi_pixels = frame[y1:y2, x1:x2]
    roi_mask = skin_mask[y1:y2, x1:x2]

    total_pixels = roi_mask.size
    skin_pixels = cv2.countNonZero(roi_mask)

    if total_pixels == 0:
        return np.nan, np.nan, np.nan, 0.0

    skin_ratio = skin_pixels / total_pixels

    # Reject regions with too few valid skin pixels
    if skin_pixels < MIN_SKIN_PIXELS or skin_ratio < MIN_SKIN_RATIO:
        return np.nan, np.nan, np.nan, skin_ratio

    # OpenCV stores image channels as BGR
    valid_pixels = roi_pixels[roi_mask > 0]

    mean_bgr = np.mean(valid_pixels, axis=0)

    # Convert channel order from BGR to RGB
    mean_r = float(mean_bgr[2])
    mean_g = float(mean_bgr[1])
    mean_b = float(mean_bgr[0])

    return mean_r, mean_g, mean_b, skin_ratio


# --------------------------------------------------
# PROCESS VIDEO FRAME BY FRAME
# --------------------------------------------------

records = []

frame_index = 0
detected_frames = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    # Timestamp derived from video frame position
    video_time = frame_index / fps

    # Detect face in the current frame
    _, faces = detector.detect(frame)

    # Prepare an output row
    record = {
        "frame": frame_index,
        "video_time_seconds": video_time,
        "face_detected": 0,
        "face_confidence": np.nan
    }

    # Initialise RGB values as missing
    for roi_name in ["forehead", "right_cheek", "left_cheek"]:

        for channel in ["r", "g", "b"]:
            record[f"{roi_name}_{channel}"] = np.nan

        record[f"{roi_name}_skin_ratio"] = 0.0

    if faces is not None and len(faces) > 0:

        # Select the highest-confidence face
        face = max(faces, key=lambda item: item[-1])

        record["face_detected"] = 1
        record["face_confidence"] = float(face[-1])

        detected_frames += 1

        # Recalculate ROIs using current facial landmarks
        rois = calculate_rois(face)

        if rois is not None:

            # Calculate skin mask from the original frame
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

            skin_mask = cv2.inRange(
                ycrcb,
                SKIN_LOWER,
                SKIN_UPPER
            )

            # Extract RGB independently from each ROI
            for roi_name, roi in rois.items():

                r, g, b, skin_ratio = extract_roi_rgb(
                    frame,
                    skin_mask,
                    roi
                )

                record[f"{roi_name}_r"] = r
                record[f"{roi_name}_g"] = g
                record[f"{roi_name}_b"] = b

                record[f"{roi_name}_skin_ratio"] = float(skin_ratio)

    records.append(record)

    frame_index += 1

    # Display progress every 100 frames
    if frame_index % 100 == 0:
        print(f"Processed {frame_index}/{total_frames} frames")

# Release the video
cap.release()

# --------------------------------------------------
# SAVE RGB SIGNALS
# --------------------------------------------------

df = pd.DataFrame(records)

df.to_csv(OUTPUT_CSV, index=False)

print("\nRGB EXTRACTION COMPLETED")
print("-" * 50)

print("Frames processed:", frame_index)
print("Faces detected:", detected_frames)

print(
    "Face detection percentage:",
    round(detected_frames / frame_index * 100, 2),
    "%"
)

print("\nValid RGB samples:")

for roi_name in ["forehead", "right_cheek", "left_cheek"]:

    valid_count = df[f"{roi_name}_r"].notna().sum()

    print(f"{roi_name}: {valid_count}/{frame_index}")

print("\nOutput saved:", OUTPUT_CSV)
print("Output shape:", df.shape)