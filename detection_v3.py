import cv2
import json
import numpy as np
from ultralytics import YOLO
import random

VIDEO_FILE = "game.mp4"
CALIBRATION_FILE = "calibration.json"

# --------------------------------------------------
# COLOR CLASSIFICATION (unchanged)
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

    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
    blue_mask   = cv2.inRange(hsv, (90, 60, 60), (140, 255, 255))

    yellow_ratio = cv2.countNonZero(yellow_mask) / (torso.shape[0] * torso.shape[1])
    blue_ratio   = cv2.countNonZero(blue_mask) / (torso.shape[0] * torso.shape[1])

    if yellow_ratio > 0.12:
        return "yellow"
    if blue_ratio > 0.12:
        return "blue"

    return "goalkeeper"


# --------------------------------------------------
# LOAD CALIBRATION (FIXED)
# --------------------------------------------------

with open(CALIBRATION_FILE, "r") as f:
    calib = json.load(f)

H = np.array(calib["homography"], dtype=np.float32)

FIELD_WIDTH  = calib.get("field_width", 50)
FIELD_LENGTH = calib.get("field_length", 80)


# --------------------------------------------------
# LOAD FRAME
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise Exception(f"Cannot open {VIDEO_FILE}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

middle_frame = random.randint(frame_count // 4, (3 * frame_count) // 4)

cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

ok, frame = cap.read()

if not ok:
    raise Exception("Cannot read frame")

cap.release()


# --------------------------------------------------
# UPSCALE FOR DETECTION
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
# VISUALIZATION SETUP
# --------------------------------------------------

original_view = frame.copy()
all_detections_view = frame.copy()
filtered_view = frame.copy()


# --------------------------------------------------
# TACTICAL MAP
# --------------------------------------------------

scale = 10

FIELD_MARGIN_METERS = 3

tactical_width = FIELD_WIDTH + 2 * FIELD_MARGIN_METERS
tactical_height = FIELD_LENGTH + 2 * FIELD_MARGIN_METERS

tactical = np.ones(
    (int(tactical_height * scale),
     int(tactical_width * scale),
     3),
    dtype=np.uint8
) * 255

field_x_offset = FIELD_MARGIN_METERS * scale
field_y_offset = FIELD_MARGIN_METERS * scale

cv2.rectangle(
    tactical,
    (int(field_x_offset), int(field_y_offset)),
    (int((FIELD_MARGIN_METERS + FIELD_WIDTH) * scale),
     int((FIELD_MARGIN_METERS + FIELD_LENGTH) * scale)),
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

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # undo upscale
    x1 //= 2
    y1 //= 2
    x2 //= 2
    y2 //= 2


    # ----------------------------------
    # FOOT POINT (IMAGE SPACE)
    # ----------------------------------

    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)


    # ----------------------------------
    # PROJECT TO FIELD SPACE (CRITICAL FIX)
    # ----------------------------------

    pt = np.array([[[float(foot_x), float(foot_y)]]], dtype=np.float32)

    projected = cv2.perspectiveTransform(pt, H)[0][0]

    field_x, field_y = float(projected[0]), float(projected[1])


    # ----------------------------------
    # FIELD FILTER (FIXED)
    # ----------------------------------

    if not (0 <= field_x <= FIELD_WIDTH and 0 <= field_y <= FIELD_LENGTH):
        continue

    players_on_field += 1


    # ----------------------------------
    # TEAM CLASSIFICATION
    # ----------------------------------

    team = classify_player_color(frame, x1, y1, x2, y2)

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
    # FILTERED VIEW (IMAGE SPACE)
    # ----------------------------------

    cv2.rectangle(filtered_view, (x1, y1), (x2, y2), draw_color, 2)

    cv2.circle(filtered_view, (foot_x, foot_y), 4, (0, 0, 255), -1)

    cv2.putText(
        filtered_view,
        f"{players_on_field} {team}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        draw_color,
        2
    )


    # ----------------------------------
    # TACTICAL MAP (FIELD SPACE)
    # ----------------------------------

    tx = int((field_x + FIELD_MARGIN_METERS) * scale)
    ty = int((field_y + FIELD_MARGIN_METERS) * scale)

    if 0 <= tx < tactical.shape[1] and 0 <= ty < tactical.shape[0]:

        if team == "yellow":
            tactical_color = (0, 255, 255)
        elif team == "blue":
            tactical_color = (255, 0, 0)
        else:
            tactical_color = (180, 180, 180)

        cv2.circle(tactical, (tx, ty), 6, tactical_color, -1)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nDetection statistics")
print("--------------------")
print("All people detected :", total_people)
print("Players on field    :", players_on_field)
print("Yellow players      :", yellow_players)
print("Blue players        :", blue_players)
print("Goalkeepers         :", goalkeepers)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

cv2.imshow("Original Frame", original_view)
cv2.imshow("All Person Detections", all_detections_view)
cv2.imshow("Players On Field", filtered_view)
cv2.imshow("Top Down Projection", tactical)

cv2.waitKey(0)
cv2.destroyAllWindows()