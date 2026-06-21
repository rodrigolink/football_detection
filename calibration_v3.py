import cv2
import numpy as np
import json

# =====================================================
# CONFIG
# =====================================================

VIDEO_FILE = "game2.mp4"

FIELD_WIDTH = 25
FIELD_LENGTH = 45

# =====================================================
# CLICK STORAGE
# =====================================================

points = []

labels = [
    "Far goal line P1",
    "Far goal line P2",

    "Near goal line P1",
    "Near goal line P2",

    "Left sideline P1",
    "Left sideline P2",

    "Right sideline P1",
    "Right sideline P2"
]

# =====================================================
# MOUSE CALLBACK
# =====================================================

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 8:
            points.append(np.array([x, y], dtype=np.float32))
            print(f"{len(points)}: {labels[len(points)-1]} -> {x}, {y}")

# =====================================================
# LOAD FRAME
# =====================================================
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise Exception(f"Cannot open {VIDEO_FILE}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

middle_frame = frame_count // 2

cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

ok, frame = cap.read()

if not ok:
    raise Exception("Cannot read frame")

cap.release()



cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", click)

# =====================================================
# CLICK LOOP
# =====================================================

while True:
    img = frame.copy()

    # draw points
    for i, p in enumerate(points):
        cv2.circle(img, tuple(p.astype(int)), 6, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1),
                    tuple(p.astype(int) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # instructions
    if len(points) < 8:
        cv2.putText(img,
                    f"STEP {len(points)+1}/8: {labels[len(points)]}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2)
    else:
        cv2.putText(img,
                    "DONE - computing calibration...",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2)

    cv2.imshow("Calibration", img)

    key = cv2.waitKey(10)

    if len(points) == 8:
        break

    if key == 27:
        exit()

cv2.destroyAllWindows()

p = np.array(points, dtype=np.float32)

# =====================================================
# LINE FUNCTIONS
# =====================================================

def line(p1, p2):
    return np.cross([p1[0], p1[1], 1],
                    [p2[0], p2[1], 1])

def intersect(l1, l2):
    x, y, w = np.cross(l1, l2)
    return np.array([x / w, y / w])

# =====================================================
# BUILD LINES
# =====================================================

far_line   = line(p[0], p[1])
near_line  = line(p[2], p[3])
left_line  = line(p[4], p[5])
right_line = line(p[6], p[7])

# =====================================================
# INTERSECT → RECTANGLE CORNERS
# =====================================================

tl = intersect(far_line, left_line)
tr = intersect(far_line, right_line)
bl = intersect(near_line, left_line)
br = intersect(near_line, right_line)

img_rect = np.array([tl, tr, br, bl], dtype=np.float32)

# =====================================================
# FIELD MODEL (50 x 80)
# =====================================================

field_rect = np.array([
    [0, 0],
    [FIELD_WIDTH, 0],
    [FIELD_WIDTH, FIELD_LENGTH],
    [0, FIELD_LENGTH]
], dtype=np.float32)

# =====================================================
# HOMOGRAPHY (image → field)
# =====================================================

H, _ = cv2.findHomography(img_rect, field_rect)

# =====================================================
# VISIBILITY VALIDATION
# =====================================================

Hinv = np.linalg.inv(H)

scale = 20

vis_w = int(FIELD_WIDTH * scale)
vis_h = int(FIELD_LENGTH * scale)

visibility = np.ones(
    (vis_h, vis_w, 3),
    dtype=np.uint8
) * 255

# field outline
cv2.rectangle(
    visibility,
    (0, 0),
    (vis_w - 1, vis_h - 1),
    (0, 0, 0),
    2
)

img_h, img_w = frame.shape[:2]

for py in range(vis_h):

    for px in range(vis_w):

        field_x = px / scale
        field_y = py / scale

        pt_field = np.array(
            [[[field_x, field_y]]],
            dtype=np.float32
        )

        img_pt = cv2.perspectiveTransform(
            pt_field,
            Hinv
        )[0][0]

        ix = img_pt[0]
        iy = img_pt[1]

        visible = (
            0 <= ix < img_w and
            0 <= iy < img_h
        )

        if not visible:
            visibility[py, px] = (180, 180, 255)

# center line
cv2.line(
    visibility,
    (0, vis_h // 2),
    (vis_w, vis_h // 2),
    (100, 100, 100),
    1
)

# center circle
cv2.circle(
    visibility,
    (vis_w // 2, vis_h // 2),
    int(3 * scale),
    (100, 100, 100),
    1
)

# legend
cv2.putText(
    visibility,
    "RED = hidden from camera",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 255),
    2
)

cv2.imshow("Visibility Check", visibility)

print()
print("Review visibility map.")
print("Press Y to accept.")
print("Press N to recalibrate.")

while True:

    key = cv2.waitKey(0)

    if key in [ord("y"), ord("Y")]:
        break

    if key in [ord("n"), ord("N")]:
        print("Calibration rejected.")
        exit()

cv2.destroyAllWindows()
# =====================================================
# EXPORT CALIBRATION
# =====================================================

calibration = {
    "field_width": FIELD_WIDTH,
    "field_length": FIELD_LENGTH,
    "homography": H.tolist()
}

with open("calibration2.json", "w") as f:
    json.dump(calibration, f, indent=4)

print("\nSaved calibration.json")

# =====================================================
# OPTIONAL: TEST PROJECTION OF POINTS
# =====================================================

def to_field(pt):
    pt = np.array([[pt]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0][0]
    return out

test = p[0]
print("\nTest projection (first point):")
print("image:", test, "→ field:", to_field(test))