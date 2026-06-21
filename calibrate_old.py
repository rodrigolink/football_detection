import cv2
import json
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage:")
    print("python calibrate.py game.mp4")
    sys.exit()

VIDEO_FILE = sys.argv[1]

cap = cv2.VideoCapture(VIDEO_FILE)

ok, frame = cap.read()
cap.release()

if not ok:
    print("Could not read video")
    sys.exit()

display = frame.copy()

field_points = []
midfield_points = []

mode = "FIELD"

WINDOW = "Calibration"


def redraw():
    global display

    display = frame.copy()

    # Draw field polygon points
    for i, p in enumerate(field_points):

        cv2.circle(
            display,
            p,
            6,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            display,
            str(i + 1),
            (p[0] + 10, p[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    if len(field_points) > 1:
        cv2.polylines(
            display,
            [np.array(field_points)],
            False,
            (0, 255, 0),
            2
        )

    if len(field_points) == 5:
        cv2.polylines(
            display,
            [np.array(field_points)],
            True,
            (0, 255, 0),
            2
        )

    # Draw midfield line points
    for i, p in enumerate(midfield_points):

        cv2.circle(
            display,
            p,
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            f"M{i+1}",
            (p[0] + 10, p[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    if len(midfield_points) == 2:

        cv2.line(
            display,
            midfield_points[0],
            midfield_points[1],
            (0, 0, 255),
            2
        )

    if mode == "FIELD":
        text = (
            f"FIELD POLYGON: "
            f"{len(field_points)}/5 points"
        )
    else:
        text = (
            f"MIDFIELD LINE: "
            f"{len(midfield_points)}/2 points"
        )

    cv2.putText(
        display,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


def save_config():

    data = {
        "field_polygon": field_points,
        "midfield_line": midfield_points
    }

    with open(
        "config.json",
        "w"
    ) as f:
        json.dump(
            data,
            f,
            indent=4
        )

    print()
    print("Saved config.json")
    print(json.dumps(data, indent=4))


def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):
    global mode

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if mode == "FIELD":

        if len(field_points) < 5:

            field_points.append(
                (x, y)
            )

            if len(field_points) == 5:

                print(
                    "Field complete."
                )
                print(
                    "Now click 2 midfield points."
                )

                mode = "MIDFIELD"

    elif mode == "MIDFIELD":

        if len(midfield_points) < 2:

            midfield_points.append(
                (x, y)
            )

            if len(midfield_points) == 2:

                save_config()

    redraw()


cv2.namedWindow(WINDOW)

cv2.setMouseCallback(
    WINDOW,
    mouse_callback
)

redraw()

print()
print("Calibration")
print()
print("Click 5 field polygon points.")
print("Then click 2 midfield line points.")
print()
print("R = reset current stage")
print("ESC = quit")

while True:

    cv2.imshow(
        WINDOW,
        display
    )

    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        break

    elif key == ord("r"):

        if mode == "FIELD":
            field_points.clear()

        else:
            midfield_points.clear()

        redraw()

cv2.destroyAllWindows()