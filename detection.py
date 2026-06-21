import cv2
import json
import numpy as np
from ultralytics import YOLO
import random

VIDEO_FILE = "game.mp4"
CALIBRATION_FILE = "calibration.json"

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

    hsv = cv2.cvtColor(
        torso,
        cv2.COLOR_BGR2HSV
    )

    # Yellow bibs

    yellow_mask = cv2.inRange(
        hsv,
        (15, 80, 80),
        (40, 255, 255)
    )

    # Blue bibs

    blue_mask = cv2.inRange(
        hsv,
        (90, 60, 60),
        (140, 255, 255)
    )

    yellow_pixels = cv2.countNonZero(
        yellow_mask
    )

    blue_pixels = cv2.countNonZero(
        blue_mask
    )

    total_pixels = torso.shape[0] * torso.shape[1]

    yellow_ratio = yellow_pixels / total_pixels
    blue_ratio = blue_pixels / total_pixels

    if yellow_ratio > 0.12:
        return "yellow"

    if blue_ratio > 0.12:
        return "blue"

    return "goalkeeper"


# --------------------------------------------------
# LOAD CALIBRATION
# --------------------------------------------------

with open(CALIBRATION_FILE, "r") as f:
    calib = json.load(f)

field_polygon = np.array(
    calib["field_polygon"],
    dtype=np.int32
)

image_pts = np.array(
    calib["homography_points"],
    dtype=np.float32
)

field_pts = np.array([
    [10, 10],
    [40, 10],
    [40, 20],
    [10, 20],

    [10, 60],
    [40, 60],
    [40, 70],
    [10, 70]
], dtype=np.float32)

H, _ = cv2.findHomography(
    image_pts,
    field_pts
)

# --------------------------------------------------
# LOAD MIDDLE FRAME
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise Exception(
        f"Cannot open {VIDEO_FILE}"
    )

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

middle_frame = random.randint(
    frame_count // 4,
    (3 * frame_count) // 4
)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    middle_frame
)

ok, frame = cap.read()

if not ok:
    raise Exception(
        "Cannot read frame"
    )

cap.release()

# --------------------------------------------------
# UPSCALE FOR SMALL PLAYERS
# --------------------------------------------------

big_frame = cv2.resize(
    frame,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)

# --------------------------------------------------
# YOLO
# --------------------------------------------------

print("Loading YOLO...")

model = YOLO("yolo11m.pt")

print("Running detection...")

results = model(
    big_frame,
    imgsz=1920,
    conf=0.10,
    verbose=False
)

# --------------------------------------------------
# VIEWS
# --------------------------------------------------

original_view = frame.copy()
all_detections_view = frame.copy()
filtered_view = frame.copy()

cv2.polylines(
    filtered_view,
    [field_polygon],
    True,
    (0, 255, 255),
    2
)

# --------------------------------------------------
# TACTICAL MAP
# --------------------------------------------------

scale = 10

field_width = 55
field_height = 75

FIELD_MARGIN_METERS = 3

tactical_width = field_width + 2 * FIELD_MARGIN_METERS
tactical_height = field_height + 2 * FIELD_MARGIN_METERS

tactical = np.ones(
    (
        int(tactical_height * scale),
        int(tactical_width * scale),
        3
    ),
    dtype=np.uint8
) * 255

field_x1 = int(FIELD_MARGIN_METERS * scale)
field_y1 = int(FIELD_MARGIN_METERS * scale)

field_x2 = int((FIELD_MARGIN_METERS + field_width) * scale)
field_y2 = int((FIELD_MARGIN_METERS + field_height) * scale)

cv2.rectangle(
    tactical,
    (field_x1, field_y1),
    (field_x2, field_y2),
    (0, 0, 0),
    2
)

# --------------------------------------------------
# STATS
# --------------------------------------------------

total_people = 0
players_on_field = 0

yellow_players = 0
blue_players = 0
goalkeepers = 0

# --------------------------------------------------
# PROCESS DETECTIONS
# --------------------------------------------------

for box in results[0].boxes:

    cls = int(box.cls[0])

    if cls != 0:
        continue

    total_people += 1

    conf = float(box.conf[0])

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    # Convert back from 2x frame

    x1 //= 2
    y1 //= 2
    x2 //= 2
    y2 //= 2

    # ----------------------------------
    # ALL DETECTIONS WINDOW
    # ----------------------------------

    cv2.rectangle(
        all_detections_view,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        all_detections_view,
        f"{conf:.2f}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 0),
        1
    )

    # ----------------------------------
    # FOOT POINT
    # ----------------------------------
    foot_x = int(
        (x1 + x2) / 2
    )

    foot_y = int(y2)

    distance = cv2.pointPolygonTest(
        field_polygon,
        (foot_x, foot_y),
        True
    )
    FIELD_MARGIN = 40


    if distance < -FIELD_MARGIN:
        continue

    players_on_field += 1

    # ----------------------------------
    # TEAM CLASSIFICATION
    # ----------------------------------

    team = classify_player_color(
        frame,
        x1,
        y1,
        x2,
        y2
    )

    if team == "yellow":
        yellow_players += 1
        draw_color = (0, 255, 255)

    elif team == "blue":
        blue_players += 1
        draw_color = (255, 0, 0)

    else:
        goalkeepers += 1
        draw_color = (180, 180, 180)

    # ----------------------------------
    # FILTERED VIEW
    # ----------------------------------

    cv2.rectangle(
        filtered_view,
        (x1, y1),
        (x2, y2),
        draw_color,
        2
    )

    cv2.circle(
        filtered_view,
        (foot_x, foot_y),
        4,
        (0, 0, 255),
        -1
    )

    label = f"{players_on_field} {team}"

    cv2.putText(
        filtered_view,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        draw_color,
        2
    )

    # ----------------------------------
    # PROJECT TO FIELD
    # ----------------------------------

    pt = np.array(
        [[[foot_x, foot_y]]],
        dtype=np.float32
    )

    projected = cv2.perspectiveTransform(
        pt,
        H
    )


    tx = int(
        (projected[0][0][0] + FIELD_MARGIN_METERS) * scale
    )

    ty = int(
        (projected[0][0][1] + FIELD_MARGIN_METERS) * scale
    )

    if (
        0 <= tx < tactical.shape[1]
        and
        0 <= ty < tactical.shape[0]
    ):

        if team == "yellow":
            tactical_color = (0, 255, 255)

        elif team == "blue":
            tactical_color = (255, 0, 0)

        else:
            tactical_color = (180, 180, 180)

        cv2.circle(
            tactical,
            (tx, ty),
            6,
            tactical_color,
            -1
        )

        cv2.putText(
            tactical,
            str(players_on_field),
            (tx + 6, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("Detection statistics")
print("--------------------")
print("All people detected :", total_people)
print("Players on field    :", players_on_field)
print()
print("Yellow players      :", yellow_players)
print("Blue players        :", blue_players)
print("Goalkeepers         :", goalkeepers)
print()

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

cv2.imshow(
    "Original Frame",
    original_view
)

cv2.imshow(
    "All Person Detections",
    all_detections_view
)

cv2.imshow(
    "Players On Field",
    filtered_view
)

cv2.imshow(
    "Top Down Projection",
    tactical
)

cv2.waitKey(0)
cv2.destroyAllWindows()