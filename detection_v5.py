import cv2
import json
import numpy as np
from ultralytics import YOLO
import sys
import csv

# ==================================================
# CONFIG
# ==================================================

VIDEO_FILE = "game.mp4"
CALIBRATION_FILE = "calibration.json"

FRAME_SKIP = 15

# ==================================================
# OUTPUT OPTIONS
# ==================================================

EXPORT_VIDEO = False
OUTPUT_VIDEO = "tactical_output.mp4"

EXPORT_CSV = True
CSV_FILE = "player_positions.csv"

# ==================================================
# MATCH COLORS
# ==================================================

MATCH_TYPE = "blue_yellow"
# MATCH_TYPE = "orange_black"

# ==================================================
# COLOR CLASSIFICATION
# ==================================================

def classify_player_color(frame, x1, y1, x2, y2):

    h = y2 - y1
    torso_y2 = y1 + int(h * 0.55)

    torso = frame[
        max(0, y1):max(0, torso_y2),
        max(0, x1):max(0, x2)
    ]

    if torso.size == 0:
        return "other"

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    area = torso.shape[0] * torso.shape[1]

    # ----------------------------------------------
    # BLUE / YELLOW
    # ----------------------------------------------

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
        blue_ratio = cv2.countNonZero(blue_mask) / area

        if yellow_ratio > 0.12:
            return "yellow"

        if blue_ratio > 0.12:
            return "blue"

        return "other"

    # ----------------------------------------------
    # ORANGE / BLACK
    # ----------------------------------------------

    orange_mask = cv2.inRange(
        hsv,
        (5, 100, 100),
        (25, 255, 255)
    )

    orange_ratio = cv2.countNonZero(orange_mask) / area

    if orange_ratio > 0.10:
        return "orange"

    return "black"

# ==================================================
# LOAD CALIBRATION
# ==================================================

with open(CALIBRATION_FILE, "r") as f:
    calib = json.load(f)

H = np.array(
    calib["homography"],
    dtype=np.float32
)

FIELD_WIDTH = calib.get("field_width", 25)
FIELD_LENGTH = calib.get("field_length", 45)

# ==================================================
# LOAD MODEL
# ==================================================

print("Loading YOLO...")
model = YOLO("yolo11m.pt")

# ==================================================
# VIDEO INPUT
# ==================================================

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise Exception("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

# ==================================================
# CSV OUTPUT
# ==================================================

csv_file = None
csv_writer = None

if EXPORT_CSV:

    csv_file = open(
        CSV_FILE,
        "w",
        newline=""
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "seconds",
        "frame",
        "team",
        "x",
        "y",
        "total_players"
    ])

# ==================================================
# VIDEO OUTPUT
# ==================================================

scale = 20
FIELD_MARGIN = 3

out_w = int(
    (FIELD_WIDTH + 2 * FIELD_MARGIN)
    * scale
)

out_h = int(
    (FIELD_LENGTH + 2 * FIELD_MARGIN)
    * scale
)

out = None

if EXPORT_VIDEO:

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / FRAME_SKIP,
        (out_w, out_h)
    )

# ==================================================
# FIELD CANVAS
# ==================================================

def create_canvas():

    canvas = np.ones(
        (out_h, out_w, 3),
        dtype=np.uint8
    ) * 255

    x1 = int(FIELD_MARGIN * scale)
    y1 = int(FIELD_MARGIN * scale)

    x2 = int(
        (FIELD_MARGIN + FIELD_WIDTH)
        * scale
    )

    y2 = int(
        (FIELD_MARGIN + FIELD_LENGTH)
        * scale
    )

    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        2
    )

    return canvas

# ==================================================
# PROGRESS BAR
# ==================================================

def print_progress(i, total):

    p = i / total

    bar = (
        "█" * int(40 * p)
        + "-" * (40 - int(40 * p))
    )

    sys.stdout.write(
        f"\rProcessing |{bar}| "
        f"{p*100:.1f}% "
        f"({i}/{total})"
    )

    sys.stdout.flush()

# ==================================================
# MAIN LOOP
# ==================================================

frame_idx = 0

print(
    f"Processing video "
    f"(frame skip = {FRAME_SKIP})..."
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    if frame_idx % FRAME_SKIP != 0:
        continue

    print_progress(
        frame_idx,
        total_frames
    )

    big_frame = cv2.resize(
        frame,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    results = model(
        big_frame,
        imgsz=1920,
        conf=0.10,
        verbose=False
    )

    projected_players = []

    # ----------------------------------------------
    # FIRST PASS
    # ----------------------------------------------

    for box in results[0].boxes:

        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        x1 //= 2
        y1 //= 2
        x2 //= 2
        y2 //= 2

        foot_x = int(
            (x1 + x2) / 2
        )

        foot_y = int(y2)

        pt = np.array(
            [[[float(foot_x), float(foot_y)]]],
            dtype=np.float32
        )

        projected = cv2.perspectiveTransform(
            pt,
            H
        )[0][0]

        field_x = float(projected[0])
        field_y = float(projected[1])

        if not (
            0 <= field_x <= FIELD_WIDTH
            and
            0 <= field_y <= FIELD_LENGTH
        ):
            continue

        team = classify_player_color(
            frame,
            x1,
            y1,
            x2,
            y2
        )

        projected_players.append(
            (
                field_x,
                field_y,
                team
            )
        )

    total_players_detected = len(
        projected_players
    )

    # ----------------------------------------------
    # DRAW VIDEO
    # ----------------------------------------------

    tactical = None

    if EXPORT_VIDEO:
        tactical = create_canvas()

    # ----------------------------------------------
    # SECOND PASS
    # ----------------------------------------------

    for field_x, field_y, team in projected_players:

        if EXPORT_CSV:

            csv_writer.writerow([
                round(
                    frame_idx / fps,
                    3
                ),
                frame_idx,
                team,
                round(field_x, 2),
                round(field_y, 2),
                total_players_detected
            ])

        if EXPORT_VIDEO:

            tx = int(
                (field_x + FIELD_MARGIN)
                * scale
            )

            ty = int(
                (field_y + FIELD_MARGIN)
                * scale
            )

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

            cv2.circle(
                tactical,
                (tx, ty),
                5,
                color,
                -1
            )

    if EXPORT_VIDEO:
        out.write(tactical)

# ==================================================
# CLEANUP
# ==================================================

cap.release()

if out is not None:
    out.release()

if csv_file is not None:
    csv_file.close()

print("\nDone.")

if EXPORT_CSV:
    print(
        f"CSV saved to {CSV_FILE}"
    )

if EXPORT_VIDEO:
    print(
        f"Video saved to {OUTPUT_VIDEO}"
    )