import cv2
import csv
import json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import sys
import os

# ==================================================
# CONFIGURATION
# ==================================================

VIDEO_FILE = "game.mp4"
CALIBRATION_FILE = "calibration.json"

# Match type:
# "blue_yellow"
# "orange_black"
MATCH_TYPE = "blue_yellow"

# Tracking cadence
TRACK_EVERY_N_FRAMES = 3
EXPORT_EVERY_N_FRAMES = 15

# Outputs
GENERATE_VIDEO = True
GENERATE_CSV = True

# Team voting
TEAM_LOCK_VOTES = 10

# Tracking settings
TRACK_TIMEOUT_SECONDS = 10

# YOLO settings
YOLO_MODEL = "yolo11m.pt"
CONFIDENCE = 0.10
IMGSZ = 1920

# Speed sanity check (m/s)
MAX_PLAYER_SPEED = 12.0

# Tactical view
FIELD_MARGIN = 3
SCALE = 20

# ==================================================
# TEAM COLOR CLASSIFICATION
# ==================================================

def classify_player_color(frame, x1, y1, x2, y2):

    h = y2 - y1

    torso_y2 = y1 + int(h * 0.55)

    torso = frame[
        max(0, y1):max(0, torso_y2),
        max(0, x1):max(0, x2)
    ]

    if torso.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    area = torso.shape[0] * torso.shape[1]

    # ------------------------------------------
    # BLUE vs YELLOW
    # ------------------------------------------

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

        return "unknown"

    # ------------------------------------------
    # ORANGE vs BLACK
    # ------------------------------------------

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
# PROGRESS BAR
# ==================================================

def print_progress(current, total):

    p = current / total

    bar = (
        "█" * int(40 * p)
        + "-" * (40 - int(40 * p))
    )

    sys.stdout.write(
        f"\rProcessing |{bar}| {p*100:.1f}%"
    )

    sys.stdout.flush()

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
# LOAD VIDEO
# ==================================================

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise Exception(
        f"Cannot open video: {VIDEO_FILE}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

TRACK_TIMEOUT_FRAMES = int(
    fps * TRACK_TIMEOUT_SECONDS
)

# ==================================================
# LOAD YOLO
# ==================================================

print("Loading YOLO...")

model = YOLO(YOLO_MODEL)

# ==================================================
# CSV OUTPUT
# ==================================================

csv_writer = None
csv_file = None

if GENERATE_CSV:

    csv_file = open(
        "player_tracks.csv",
        "w",
        newline=""
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "time_sec",
        "frame",
        "player_id",
        "team",
        "team_confidence",
        "x",
        "y"
    ])

# ==================================================
# TACTICAL VIDEO OUTPUT
# ==================================================

video_out = None

out_w = int(
    (FIELD_WIDTH + FIELD_MARGIN * 2)
    * SCALE
)

out_h = int(
    (FIELD_LENGTH + FIELD_MARGIN * 2)
    * SCALE
)

if GENERATE_VIDEO:

    video_out = cv2.VideoWriter(
        "tactical_output2.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / TRACK_EVERY_N_FRAMES,
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

    x1 = int(FIELD_MARGIN * SCALE)
    y1 = int(FIELD_MARGIN * SCALE)

    x2 = int(
        (FIELD_MARGIN + FIELD_WIDTH)
        * SCALE
    )

    y2 = int(
        (FIELD_MARGIN + FIELD_LENGTH)
        * SCALE
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
# PLAYER STRUCTURES
# ==================================================

next_player_number = 1

tracker_to_player = {}

player_team_votes = defaultdict(
    lambda: defaultdict(int)
)

locked_team = {}

last_seen_frame = {}

player_positions = {}

retired_players = set()

# ==================================================
# TEAM CONFIDENCE
# ==================================================

def get_team_confidence(track_id):

    votes = player_team_votes[track_id]

    total = sum(votes.values())

    if total == 0:
        return 0.0

    winner = max(votes.values())

    return winner / total

# ==================================================
# PLAYER ID CREATION
# ==================================================

def create_player_id():

    global next_player_number

    pid = f"player{next_player_number:03d}"

    next_player_number += 1

    return pid
    
    
tracker_last_seen = {}
player_tracker = {}

# ==================================================
# MAIN LOOP
# ==================================================

frame_idx = 0

print("Processing video...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    print_progress(frame_idx, total_frames)

    # ------------------------------------------
    # Only track every N frames
    # ------------------------------------------

    if frame_idx % TRACK_EVERY_N_FRAMES != 0:
        continue

    current_time = frame_idx / fps

    # ------------------------------------------
    # Upscale for detection
    # ------------------------------------------

    big_frame = cv2.resize(
        frame,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # ------------------------------------------
    # YOLO + ByteTrack
    # ------------------------------------------

    results = model.track(
        big_frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        verbose=False
    )

    tactical = None

    if GENERATE_VIDEO:
        tactical = create_canvas()

    # ------------------------------------------
    # Tracks seen this frame
    # ------------------------------------------

    seen_tracks = set()

    # ------------------------------------------
    # Safety
    # ------------------------------------------

    if len(results) == 0:
        continue

    if results[0].boxes is None:
        continue

    # ------------------------------------------
    # Process tracks
    # ------------------------------------------

    for box in results[0].boxes:

        # person class only
        if int(box.cls[0]) != 0:
            continue

        # ByteTrack ID required
        if box.id is None:
            continue

        tracker_id = int(box.id[0])

        seen_tracks.add(tracker_id)
        tracker_last_seen[tracker_id] = frame_idx

        # --------------------------------------
        # Bounding box
        # --------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # undo upscale
        x1 //= 2
        y1 //= 2
        x2 //= 2
        y2 //= 2

        # --------------------------------------
        # Create player ID if needed
        # --------------------------------------

        if tracker_id not in tracker_to_player:

            tracker_to_player[tracker_id] = (
                create_player_id()
            )

        player_id = tracker_to_player[tracker_id]
        player_tracker[player_id] = tracker_id

        # --------------------------------------
        # Foot point
        # --------------------------------------

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

        # --------------------------------------
        # Ignore outside field
        # --------------------------------------

        if not (
            0 <= field_x <= FIELD_WIDTH
            and
            0 <= field_y <= FIELD_LENGTH
        ):
            continue

        # --------------------------------------
        # Speed sanity check
        # --------------------------------------

        if player_id in player_positions:

            last_x, last_y, last_frame = (
                player_positions[player_id]
            )

            dt = (
                frame_idx - last_frame
            ) / fps

            if dt > 0:

                distance = np.sqrt(
                    (field_x - last_x) ** 2 +
                    (field_y - last_y) ** 2
                )

                speed = distance / dt

                if speed > MAX_PLAYER_SPEED:
                    continue

        # --------------------------------------
        # Save position
        # --------------------------------------

        player_positions[player_id] = (
            field_x,
            field_y,
            frame_idx
        )

        last_seen_frame[player_id] = (
            frame_idx
        )

        # --------------------------------------
        # Team voting
        # --------------------------------------

        observed_team = classify_player_color(
            frame,
            x1,
            y1,
            x2,
            y2
        )

        if player_id not in locked_team:

            if observed_team != "unknown":

                player_team_votes[player_id][
                    observed_team
                ] += 1

                total_votes = sum(
                    player_team_votes[player_id].values()
                )

                if total_votes >= TEAM_LOCK_VOTES:

                    winner = max(
                        player_team_votes[player_id],
                        key=lambda k:
                        player_team_votes[player_id][k]
                    )

                    locked_team[player_id] = winner

        # --------------------------------------
        # Final team
        # --------------------------------------

        if player_id in locked_team:

            team = locked_team[player_id]

        else:

            team = observed_team

        # --------------------------------------
        # Team confidence
        # --------------------------------------

        team_confidence = (
            get_team_confidence(player_id)
        )

        # --------------------------------------
        # Export CSV
        # --------------------------------------

        if (
            GENERATE_CSV
            and
            frame_idx % EXPORT_EVERY_N_FRAMES == 0
        ):

            csv_writer.writerow([
                round(current_time, 2),
                frame_idx,
                player_id,
                team,
                round(team_confidence, 3),
                round(field_x, 2),
                round(field_y, 2)
            ])

        # --------------------------------------
        # Tactical coordinates
        # --------------------------------------

        if GENERATE_VIDEO:

            tx = int(
                (field_x + FIELD_MARGIN)
                * SCALE
            )

            ty = int(
                (field_y + FIELD_MARGIN)
                * SCALE
            )

            # ------------------------------
            # Team color
            # ------------------------------

            if team == "yellow":
                color = (0, 255, 255)

            elif team == "blue":
                color = (255, 0, 0)

            elif team == "orange":
                color = (0, 165, 255)

            elif team == "black":
                color = (0, 0, 0)

            else:
                color = (150, 150, 150)

            # ------------------------------
            # Draw player
            # ------------------------------

            cv2.circle(
                tactical,
                (tx, ty),
                5,
                color,
                -1
            )

            cv2.putText(
                tactical,
                player_id,
                (tx + 6, ty - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA
            )

    # ------------------------------------------
    # Retire inactive players
    # ------------------------------------------

for player_id in list(last_seen_frame.keys()):

    missing = (
        frame_idx
        - last_seen_frame[player_id]
    )

    if missing <= TRACK_TIMEOUT_FRAMES:
        continue

    if player_id in retired_players:
        continue

    retired_players.add(player_id)

    if player_id in player_tracker:

        old_tracker = player_tracker[player_id]

        if old_tracker in tracker_to_player:
            del tracker_to_player[old_tracker]

    # ------------------------------------------
    # Write tactical frame
    # ------------------------------------------

    if GENERATE_VIDEO:

        video_out.write(tactical)
        
    print("\n")

# ==================================================
# CLEANUP
# ==================================================

cap.release()

if GENERATE_VIDEO and video_out is not None:
    video_out.release()

if GENERATE_CSV and csv_file is not None:
    csv_file.close()

# ==================================================
# SUMMARY
# ==================================================

print("===================================")
print("Processing complete")
print("===================================")

print(f"Frames processed: {frame_idx}")

print(
    f"Players created: "
    f"{next_player_number - 1}"
)

print(
    f"Players retired: "
    f"{len(retired_players)}"
)

active_players = (
    next_player_number - 1
    - len(retired_players)
)

print(
    f"Players still active: "
    f"{active_players}"
)

if GENERATE_CSV:

    print(
        "CSV saved: "
        "player_tracks.csv"
    )

if GENERATE_VIDEO:

    print(
        "Video saved: "
        "tactical_output2.mp4"
    )

print("Done.")