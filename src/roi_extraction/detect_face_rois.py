import cv2
import json
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "datasets/raw/subject3/vid.avi"
MODEL_PATH = ROOT / "models/yunet/face_detection_yunet_2023mar.onnx"

OUTPUT_DIR = ROOT / "outputs/plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_OUTPUT = OUTPUT_DIR / "subject3_roi_verification.jpg"
JSON_OUTPUT = OUTPUT_DIR / "subject3_roi_verification.json"

# --------------------------------------------------
# CLINICAL COLOUR PALETTE (BGR)
# --------------------------------------------------

COLORS = {
    "Forehead": (55, 190, 105),       # Medical green
    "Right Cheek": (190, 115, 95),   # Muted blue
    "Left Cheek": (175, 105, 175)    # Muted purple
}

HEADER_COLOR = (35, 65, 55)
WHITE = (245, 250, 248)

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
# READ VIDEO FRAME
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Cannot open subject3 video.")

fps = cap.get(cv2.CAP_PROP_FPS)

# Select a frame at approximately 10 seconds
cap.set(cv2.CAP_PROP_POS_MSEC, 10000)

success, original_frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Cannot read the selected frame.")

height, width = original_frame.shape[:2]
detector.setInputSize((width, height))

# --------------------------------------------------
# FACE DETECTION
# --------------------------------------------------

_, faces = detector.detect(original_frame)

if faces is None or len(faces) == 0:
    raise RuntimeError("No face detected.")

# Select the face with the highest confidence
face = max(faces, key=lambda item: item[-1])

x, y, w, h = face[:4].astype(int)

# YuNet landmark order:
# Right eye, left eye, nose, right mouth corner, left mouth corner
landmarks = face[4:14].reshape(5, 2)

right_eye, left_eye, nose, right_mouth, left_mouth = landmarks

eye_mid = (right_eye + left_eye) / 2
mouth_mid = (right_mouth + left_mouth) / 2

eye_distance = np.linalg.norm(right_eye - left_eye)

if eye_distance < 10:
    raise RuntimeError("Unreliable facial landmark detection.")

# --------------------------------------------------
# DEFINE FACE-BOUNDARY LIMITS
# --------------------------------------------------

face_left = max(0, x)
face_top = max(0, y)
face_right = min(width, x + w)
face_bottom = min(height, y + h)

# Create a rectangle and keep it inside the face boundary
def create_roi(x1, y1, x2, y2):

    x1 = max(face_left, int(round(x1)))
    y1 = max(face_top, int(round(y1)))
    x2 = min(face_right, int(round(x2)))
    y2 = min(face_bottom, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI coordinates.")

    return (x1, y1, x2, y2)

# --------------------------------------------------
# FOREHEAD ROI
# --------------------------------------------------

forehead = create_roi(
    eye_mid[0] - 0.55 * eye_distance,
    eye_mid[1] - 0.95 * eye_distance,
    eye_mid[0] + 0.55 * eye_distance,
    eye_mid[1] - 0.20 * eye_distance
)

# --------------------------------------------------
# CHEEK ROIS
# --------------------------------------------------

# Cheek ROIs are positioned outward from the nose,
# below the eyes and above the mouth.

cheek_top = eye_mid[1] + 0.40 * eye_distance
cheek_bottom = mouth_mid[1] - 0.15 * eye_distance

cheek_width = 0.55 * eye_distance

# Subject's right cheek (appears on the left of the image)
right_cheek = create_roi(
    right_eye[0] - cheek_width,
    cheek_top,
    right_eye[0] + 0.05 * eye_distance,
    cheek_bottom
)

# Subject's left cheek (appears on the right of the image)
left_cheek = create_roi(
    left_eye[0] - 0.05 * eye_distance,
    cheek_top,
    left_eye[0] + cheek_width,
    cheek_bottom
)

rois = {
    "Forehead": forehead,
    "Right Cheek": right_cheek,
    "Left Cheek": left_cheek
}

# --------------------------------------------------
# SKIN MASK ON ORIGINAL FRAME
# --------------------------------------------------

# Calculate skin mask before adding any annotations
ycrcb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)

skin_mask = cv2.inRange(
    ycrcb,
    np.array([0, 133, 77], dtype=np.uint8),
    np.array([255, 173, 127], dtype=np.uint8)
)

# --------------------------------------------------
# VISUALISATION
# --------------------------------------------------

# Work on a separate copy of the original image
display_frame = original_frame.copy()
overlay = display_frame.copy()

roi_results = {}

for name, (x1, y1, x2, y2) in rois.items():

    # Draw translucent ROI fill
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        COLORS[name],
        -1
    )

    # Calculate skin percentage from the original mask
    roi_mask = skin_mask[y1:y2, x1:x2]

    total_pixels = roi_mask.size
    skin_pixels = cv2.countNonZero(roi_mask)

    skin_percentage = (
        100 * skin_pixels / total_pixels
        if total_pixels else 0
    )

    roi_results[name] = {
        "coordinates": [int(x1), int(y1), int(x2), int(y2)],
        "total_pixels": int(total_pixels),
        "skin_pixels": int(skin_pixels),
        "skin_percentage": float(round(skin_percentage, 2))
    }

# Blend translucent ROI regions with the original frame
display_frame = cv2.addWeighted(
    overlay, 0.22,
    display_frame, 0.78,
    0
)

# Draw face boundary
cv2.rectangle(
    display_frame,
    (face_left, face_top),
    (face_right, face_bottom),
    WHITE,
    1
)

# Draw ROI outlines and labels
for name, (x1, y1, x2, y2) in rois.items():

    cv2.rectangle(
        display_frame,
        (x1, y1),
        (x2, y2),
        COLORS[name],
        2
    )

    # Put compact labels inside each ROI
    cv2.putText(
        display_frame,
        name,
        (x1 + 3, y1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLORS[name],
        1,
        cv2.LINE_AA
    )

# --------------------------------------------------
# ADD PROFESSIONAL HEADER
# --------------------------------------------------

header = np.full((55, width, 3), HEADER_COLOR, dtype=np.uint8)

cv2.putText(
    header,
    "ROI VERIFICATION | SUBJECT 3",
    (15, 24),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.58,
    WHITE,
    1,
    cv2.LINE_AA
)

cv2.putText(
    header,
    "rPPG - UBFC DATASET",
    (15, 44),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.35,
    (190, 215, 205),
    1,
    cv2.LINE_AA
)

# --------------------------------------------------
# ADD LEGEND AND FOOTER
# --------------------------------------------------

footer = np.full((65, width, 3), HEADER_COLOR, dtype=np.uint8)

legend_items = [
    ("Forehead", COLORS["Forehead"]),
    ("Right Cheek", COLORS["Right Cheek"]),
    ("Left Cheek", COLORS["Left Cheek"])
]

positions = [12, 215, 430]

for (label, color), position in zip(legend_items, positions):

    cv2.rectangle(
        footer,
        (position, 13),
        (position + 15, 28),
        color,
        -1
    )

    cv2.putText(
        footer,
        label,
        (position + 22, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA
    )

cv2.putText(
    footer,
    "Subject 3 | Face ROI assessment",
    (12, 52),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.35,
    (190, 215, 205),
    1,
    cv2.LINE_AA
)

# Combine header, video frame and footer
final_image = np.vstack([
    header,
    display_frame,
    footer
])

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------

if not cv2.imwrite(str(IMAGE_OUTPUT), final_image):
    raise RuntimeError("Could not save verification image.")

with open(JSON_OUTPUT, "w") as file:
    json.dump(roi_results, file, indent=4)

# --------------------------------------------------
# PRINT ROI REPORT
# --------------------------------------------------

print("\nSUBJECT 3 - ROI VERIFICATION")
print("-" * 45)

for name, result in roi_results.items():

    print(f"\n{name}")
    print("Coordinates:", result["coordinates"])
    print("Total pixels:", result["total_pixels"])
    print("Skin percentage:", result["skin_percentage"], "%")

print("\nImage saved:", IMAGE_OUTPUT)
print("Measurements saved:", JSON_OUTPUT)