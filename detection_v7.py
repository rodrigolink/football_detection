import cv2
import csv
import json
import numpy as np
import sys
import time

from ultralytics import YOLO
from collections import defaultdict

# ==================================================
# CONFIGURATION
# ==================================================

VIDEO_FILE = "game.mp4"
CALIBRATION_FILE = "calibration.json"

# Match type:
# "blue_yellow"
# "orange_black"

MATCH_TYPE = "blue_yellow"

# ------------------------------------------
# Tracking cadence
# ------------------------------------------

TRACK_EVERY_N_FRAMES = 3
EXPORT_EVERY_N_FRAMES = 15

# ------------------------------------------
# Outputs
# ------------------------------------------

GENERATE_VIDEO = True
GENERATE_CSV = True

# ------------------------------------------
# Team locking
# ------------------------------------------

TEAM_LOCK_VOTES = 10

# ------------------------------------------
# Player retirement
# ------------------------------------------

TRACK_TIMEOUT_SECONDS = 10

# ------------------------------------------
# YOLO
# ------------------------------------------

YOLO_MODEL = "yolo11m.pt"

CONFIDENCE = 0.10
IMGSZ = 1920

# ------------------------------------------
# Sanity filters
# ------------------------------------------

MAX_PLAYER_SPEED = 12.0  # m/s

# ------------------------------------------
# Tactical view
# ------------------------------------------

FIELD_MARGIN = 3
SCALE = 20

# ==================================================
# PERFORMANCE METRICS
# ==================================================

yolo_time = 0.0
projection_time = 0.0
color_time = 0.0
video_time = 0.0
csv_time = 0.0

# ==================================================
# COLOR CLASSIFICATION
# ==================================================

def classify_player_color_hsv(hsv_frame, x1, y1, x2, y2):

    h = y2 - y1

    torso_y2 = y1 + int(h * 0.55)

    torso = hsv_frame[
        max(0, y1):max(0, torso_y2),
        max(0, x1):max(0, x2)
    ]

    if torso.size == 0:
        return "unknown"

    area = torso.shape[0] * torso.shape[1]

    if area == 0:
        return "unknown"

    # ==================================================
    # BLUE vs YELLOW
    # ==================================================

    if MATCH_TYPE == "blue_yellow":

        yellow_mask = cv2.inRange(
            torso,
            (15, 80, 80),
            (40, 255, 255)
        )

        blue_mask = cv2.inRange(
            torso,
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

    # ==================================================
    # ORANGE vs BLACK
    # ==================================================

    orange_mask = cv2.inRange(
        torso,
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

    filled = int(40 * p)

    bar = (
        "█" * filled +
        "-" * (40 - filled)
    )

    sys.stdout.write(
        f"\rProcessing |{bar}| {p*100:.1f}%"
    )

    sys.stdout.flush()

# ==================================================
# FIELD CANVAS
# ==================================================

def create_canvas():

    canvas = np.ones(
        (OUT_H, OUT_W, 3),
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
# TEAM CONFIDENCE
# ==================================================

def get_team_confidence(player_id):

    votes = player_team_votes[player_id]

    total = sum(votes.values())

    if total == 0:
        return 0.0

    winner = max(votes.values())

    return winner / total

# ==================================================
# PLAYER ID CREATION
# ==================================================

next_player_number = 1

def create_player_id():

    global next_player_number

    pid = f"player{next_player_number:03d}"

    next_player_number += 1

    return pid

# ==================================================
# LOAD CALIBRATION
# ==================================================

with open(CALIBRATION_FILE, "r") as f:

    calib = json.load(f)

H = np.array(
    calib["homography"],
    dtype=np.float32
)

FIELD_WIDTH = calib.get(
    "field_width",
    25
)

FIELD_LENGTH = calib.get(
    "field_length",
    45
)

# ==================================================
# VIDEO INPUT
# ==================================================

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():

    raise Exception(
        f"Cannot open video: {VIDEO_FILE}"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

TRACK_TIMEOUT_FRAMES = int(
    fps * TRACK_TIMEOUT_SECONDS
)

# ==================================================
# LOAD MODEL
# ==================================================

print("Loading YOLO...")

model = YOLO(YOLO_MODEL)

# ==================================================
# CSV OUTPUT
# ==================================================

csv_file = None
csv_writer = None

if GENERATE_CSV:

    csv_file = open(
        "player_tracks3.csv",
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
# VIDEO OUTPUT
# ==================================================

OUT_W = int(
    (FIELD_WIDTH + 2 * FIELD_MARGIN)
    * SCALE
)

OUT_H = int(
    (FIELD_LENGTH + 2 * FIELD_MARGIN)
    * SCALE
)

video_out = None

if GENERATE_VIDEO:

    video_out = cv2.VideoWriter(
        "tactical_output3.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / TRACK_EVERY_N_FRAMES,
        (OUT_W, OUT_H)
    )

# ==================================================
# PLAYER DATA STRUCTURES
# ==================================================

# ByteTrack ID -> playerXXX
tracker_to_player = {}

# playerXXX -> last frame seen
last_seen_frame = {}

# playerXXX -> (x, y, frame)
player_positions = {}

# playerXXX -> votes
player_team_votes = defaultdict(
    lambda: defaultdict(int)
)

# playerXXX -> locked team
locked_team = {}

# retired players
retired_players = set()

# tracker_id -> last frame seen
tracker_last_seen = {}

# tracker_ids that have timed out
expired_trackers = set()

# ==================================================
# START
# ==================================================

print(
    f"Video FPS: {fps:.2f}"
)

print(
    f"Frames: {total_frames}"
)

print(
    f"Track every {TRACK_EVERY_N_FRAMES} frames"
)

print(
    f"Export every {EXPORT_EVERY_N_FRAMES} frames"
)

print(
    f"Timeout: {TRACK_TIMEOUT_SECONDS}s"
)

print("Starting...")

# ==================================================
# MAIN LOOP
# ==================================================

frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    print_progress(frame_idx, total_frames)

    # ------------------------------------------
    # Process only every N frames
    # ------------------------------------------

    if frame_idx % TRACK_EVERY_N_FRAMES != 0:
        continue

    current_time = frame_idx / fps

    # ------------------------------------------
    # Convert to HSV ONCE
    # ------------------------------------------

    hsv_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # ------------------------------------------
    # Upscale frame
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

    t0 = time.time()

    results = model.track(
        big_frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        verbose=False
    )

    yolo_time += (
        time.time() - t0
    )

    # ------------------------------------------
    # Tactical canvas
    # ------------------------------------------

    tactical = None

    if GENERATE_VIDEO:
        tactical = create_canvas()

    # ------------------------------------------
    # No detections?
    # ------------------------------------------

    if len(results) == 0:
        continue

    if results[0].boxes is None:
        continue

    # ------------------------------------------
    # Process tracks
    # ------------------------------------------

    for box in results[0].boxes:

        # person only
        if int(box.cls[0]) != 0:
            continue

        if box.id is None:
            continue

        tracker_id = int(box.id[0])

        # --------------------------------------
        # Tracker last seen
        # --------------------------------------

        tracker_last_seen[tracker_id] = frame_idx

        # --------------------------------------
        # Reuse protection
        # --------------------------------------

        if tracker_id in expired_trackers:

            tracker_to_player[tracker_id] = (
                create_player_id()
            )

            expired_trackers.remove(
                tracker_id
            )

        # --------------------------------------
        # Create playerXXX
        # --------------------------------------

        if tracker_id not in tracker_to_player:

            tracker_to_player[tracker_id] = (
                create_player_id()
            )

        player_id = tracker_to_player[
            tracker_id
        ]
        
	# --------------------------------------
	# Create playerXXX
	# --------------------------------------

        if tracker_id not in tracker_to_player:
            tracker_to_player[tracker_id] = create_player_id()

        player_id = tracker_to_player[tracker_id]

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
        # Foot position
        # --------------------------------------

        foot_x = int(
            (x1 + x2) / 2
        )

        foot_y = int(y2)

        # --------------------------------------
        # Homography projection
        # --------------------------------------

        t0 = time.time()

        pt = np.array(
            [[[foot_x, foot_y]]],
            dtype=np.float32
        )

        projected = cv2.perspectiveTransform(
            pt,
            H
        )[0][0]

        projection_time += (
            time.time() - t0
        )

        field_x = float(
            projected[0]
        )

        field_y = float(
            projected[1]
        )

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
        # Speed sanity filter
        # --------------------------------------

        if player_id in player_positions:

            old_x, old_y, old_frame = (
                player_positions[player_id]
            )

            dt = (
                frame_idx - old_frame
            ) / fps

            if dt > 0:

                distance = np.sqrt(
                    (field_x - old_x) ** 2
                    +
                    (field_y - old_y) ** 2
                )

                speed = distance / dt

                if speed > MAX_PLAYER_SPEED:
                    continue

        # --------------------------------------
        # Save latest position
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
        # Team assignment
        # --------------------------------------

        if player_id in locked_team:

            team = locked_team[
                player_id
            ]

        else:

            t0 = time.time()

            observed_team = (
                classify_player_color_hsv(
                    hsv_frame,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            color_time += (
                time.time() - t0
            )

            if observed_team != "unknown":

                player_team_votes[
                    player_id
                ][observed_team] += 1

                total_votes = sum(
                    player_team_votes[
                        player_id
                    ].values()
                )

                if (
                    total_votes >=
                    TEAM_LOCK_VOTES
                ):

                    winner = max(
                        player_team_votes[
                            player_id
                        ],
                        key=lambda k:
                        player_team_votes[
                            player_id
                        ][k]
                    )

                    locked_team[
                        player_id
                    ] = winner

                    team = winner

                else:

                    team = observed_team

            else:

                team = "unknown"

        # --------------------------------------
        # Team confidence
        # --------------------------------------

        team_confidence = (
            get_team_confidence(
                player_id
            )
        )

        # --------------------------------------
        # CSV export
        # --------------------------------------

        if (
            GENERATE_CSV
            and
            frame_idx %
            EXPORT_EVERY_N_FRAMES
            == 0
        ):

            t0 = time.time()

            csv_writer.writerow([
                round(
                    current_time,
                    2
                ),
                frame_idx,
                player_id,
                team,
                round(
                    team_confidence,
                    3
                ),
                round(
                    field_x,
                    2
                ),
                round(
                    field_y,
                    2
                )
            ])

            csv_time += (
                time.time() - t0
            )

        # --------------------------------------
        # Tactical rendering
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
                (tx + 5, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA
            )

    # ------------------------------------------
    # Retire players not seen recently
    # ------------------------------------------

    for tracker_id, player_id in list(
        tracker_to_player.items()
    ):

        if player_id in retired_players:
            continue

        if player_id not in last_seen_frame:
            continue

        missing = (
            frame_idx
            - last_seen_frame[player_id]
        )

        if missing <= TRACK_TIMEOUT_FRAMES:
            continue

        retired_players.add(
            player_id
        )

        expired_trackers.add(
            tracker_id
        )
    # ------------------------------------------
    # Write video frame
    # ------------------------------------------

    if GENERATE_VIDEO:

        t0 = time.time()

        video_out.write(
            tactical
        )

        video_time += (
            time.time() - t0
        )
print("\n")

# ==================================================
# CLEANUP
# ==================================================

cap.release()

if GENERATE_VIDEO:

    video_out.release()

if GENERATE_CSV:

    csv_file.close()
    
# ==================================================
# PERFORMANCE REPORT
# ==================================================

total_profiled = (
    yolo_time
    + projection_time
    + color_time
    + video_time
    + csv_time
)

print("\n================================")
print("PERFORMANCE")
print("================================")

if total_profiled > 0:

    print(
        f"YOLO:       "
        f"{100*yolo_time/total_profiled:.1f}%"
    )

    print(
        f"Projection: "
        f"{100*projection_time/total_profiled:.1f}%"
    )

    print(
        f"Color:      "
        f"{100*color_time/total_profiled:.1f}%"
    )

    print(
        f"Video:      "
        f"{100*video_time/total_profiled:.1f}%"
    )

    print(
        f"CSV:        "
        f"{100*csv_time/total_profiled:.1f}%"
    )
print("\n================================")
print("SUMMARY")
print("================================")

print(
    f"Frames processed: "
    f"{frame_idx}"
)

print(
    f"Players created: "
    f"{next_player_number - 1}"
)

print(
    f"Players retired: "
    f"{len(retired_players)}"
)

active_players = (
    next_player_number
    - 1
    - len(retired_players)
)

print(
    f"Players active: "
    f"{active_players}"
)

if GENERATE_CSV:

    print(
        "\nCSV saved:"
    )

    print(
        "player_tracks.csv"
    )

if GENERATE_VIDEO:

    print(
        "\nVideo saved:"
    )

    print(
        "tactical_output2.mp4"
    )

print("\nDone.")        