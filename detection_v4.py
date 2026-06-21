import cv2
import json
import numpy as np
from ultralytics import YOLO
import sys

VIDEO_FILE = "game2.mp4"
CALIBRATION_FILE = "calibration2.json"

# ==================================================
# FRAME SKIP OPTION (NEW)
# ==================================================

FRAME_SKIP = 60   # 👈 change this (e.g. 1 = every frame)

# ==================================================
# MATCH CONFIG
# ==================================================

#MATCH_TYPE = "blue_yellow"
MATCH_TYPE = "orange_black"

# --------------------------------------------------
# COLOR CLASSIFICATION
# --------------------------------------------------

def classify_player_color(frame, x1, y1, x2, y2):

    h = y2 - y1
    torso_y2 = y1 + int(h * 0.55)

    torso = frame[
        max(0, y1):max(0, torso_y2),
        max(0, x1):max(0, x2)
    ]

    if torso.size == 0:
        return "goalkeeper"

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    area = torso.shape[0] * torso.shape[1]

    # ==================================================
    # BLUE vs YELLOW MATCH
    # ==================================================

    if MATCH_TYPE == "blue_yellow":

        yellow_mask = cv2.inRange(
            hsv,
            (15, 80, 80),
            (40, 255, 255)
        )

        blue_mask = cv2.inRange(
            hsv,
            (90, 60, 60),
            (140, 255, 255)
        )

        yellow_ratio = cv2.countNonZero(yellow_mask) / area
        blue_ratio   = cv2.countNonZero(blue_mask) / area

        if yellow_ratio > 0.12:
            return "yellow"

        if blue_ratio > 0.12:
            return "blue"

        return "goalkeeper"

    # ==================================================
    # ORANGE vs BLACK MATCH
    # ==================================================

    orange_mask = cv2.inRange(
        hsv,
        (5, 100, 100),
        (25, 255, 255)
    )

    orange_ratio = cv2.countNonZero(orange_mask) / area

    # orange jerseys are very distinctive
    if orange_ratio > 0.10:
        return "orange"

    # everybody else = black team
    return "black"

# --------------------------------------------------
# LOAD CALIBRATION
# --------------------------------------------------

with open(CALIBRATION_FILE, "r") as f:
    calib = json.load(f)

H = np.array(calib["homography"], dtype=np.float32)

FIELD_WIDTH = calib.get("field_width", 25)
FIELD_LENGTH = calib.get("field_length", 45)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

print("Loading YOLO...")
model = YOLO("yolo11m.pt")


# --------------------------------------------------
# VIDEO INPUT
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise Exception("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# --------------------------------------------------
# OUTPUT VIDEO
# --------------------------------------------------

scale = 20
FIELD_MARGIN = 3

out_w = int((FIELD_WIDTH + 2 * FIELD_MARGIN) * scale)
out_h = int((FIELD_LENGTH + 2 * FIELD_MARGIN) * scale)

out = cv2.VideoWriter(
    "tactical_output2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (out_w, out_h)
)


# --------------------------------------------------
# FIELD CANVAS
# --------------------------------------------------

def create_canvas():
    canvas = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255

    x1 = int(FIELD_MARGIN * scale)
    y1 = int(FIELD_MARGIN * scale)
    x2 = int((FIELD_MARGIN + FIELD_WIDTH) * scale)
    y2 = int((FIELD_MARGIN + FIELD_LENGTH) * scale)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return canvas


# --------------------------------------------------
# PROGRESS BAR
# --------------------------------------------------

def print_progress(i, total):
    p = i / total
    bar = "█" * int(40 * p) + "-" * (40 - int(40 * p))
    sys.stdout.write(f"\rProcessing |{bar}| {p*100:.1f}% ({i}/{total})")
    sys.stdout.flush()


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

frame_idx = 0
processed_idx = 0

print(f"Processing video (frame skip = {FRAME_SKIP})...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # ==================================================
    # FRAME SKIP LOGIC
    # ==================================================

    if frame_idx % FRAME_SKIP != 0:
        continue

    processed_idx += 1
    print_progress(frame_idx, total_frames)

    # --------------------------------------------------
    # DETECTION
    # --------------------------------------------------

    big_frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    results = model(big_frame, imgsz=1920, conf=0.10, verbose=False)

    tactical = create_canvas()

    for box in results[0].boxes:

        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # undo upscale
        x1 //= 2
        y1 //= 2
        x2 //= 2
        y2 //= 2

        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)

        pt = np.array([[[float(foot_x), float(foot_y)]]], dtype=np.float32)

        projected = cv2.perspectiveTransform(pt, H)[0][0]

        field_x, field_y = float(projected[0]), float(projected[1])

        if not (0 <= field_x <= FIELD_WIDTH and 0 <= field_y <= FIELD_LENGTH):
            continue

        tx = int((field_x + FIELD_MARGIN) * scale)
        ty = int((field_y + FIELD_MARGIN) * scale)

        team = classify_player_color(frame, x1, y1, x2, y2)

        if team == "yellow":
            color = (0, 255, 255)

        elif team == "blue":
            color = (255, 0, 0)

        elif team == "orange":
            color = (0, 165, 255)

        elif team == "black":
            color = (0, 0, 0)

        else:
            color = (180, 180, 180)
            
        if 0 <= tx < out_w and 0 <= ty < out_h:
            cv2.circle(tactical, (tx, ty), 5, color, -1)

    out.write(tactical)

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()
out.release()

print("\nDone. Saved tactical_output.mp4")