import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans

# ==========================================
# MODULE 1: FIELD CONFIGURATION & GEOMETRY (PERMANENT VIEW)
# ==========================================
# Swapped to match your physical camera's vertical orientation
FIELD_WIDTH_M = 25  # Goal lines (horizontal across the frame)
FIELD_HEIGHT_M = 45 # Sidelines (vertical up/down the frame)
video_path = "game.mp4"

def calculate_dynamic_dimensions(frame_width, frame_height):
    """Calculates canvas dimensions to fit your screen at a 25x45 ratio."""
    max_w = frame_width * 0.8
    max_h = frame_height * 0.8
    
    scale_factor = int(min(max_w / FIELD_WIDTH_M, max_h / FIELD_HEIGHT_M))
    if scale_factor < 10: 
        scale_factor = 10
        
    return scale_factor, FIELD_WIDTH_M * scale_factor, FIELD_HEIGHT_M * scale_factor

def find_intersection(p1, p2, p3, p4):
    """Calculates the intersection of line (p1,p2) and line (p3,p4)"""
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])
    def det(a, b): return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0: return (0, 0)
    d = (det(p1, p2), det(p3, p4))
    return int(det(d, xdiff) / div), int(det(d, ydiff) / div)

def generate_blind_spot_map(H, canvas_width, canvas_height, frame_width, frame_height):
    """Generates the vertical 25x45m field map with correctly oriented markings."""
    pitch_map = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    pitch_map[:] = (50, 150, 50) 
    
    H_inv = np.linalg.inv(H)
    x_coords, y_coords = np.meshgrid(np.arange(canvas_width), np.arange(canvas_height))
    canvas_pixels = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).astype(np.float32)
    canvas_pixels = np.expand_dims(canvas_pixels, axis=1)
    
    camera_mapped = cv2.perspectiveTransform(canvas_pixels, H_inv).squeeze(axis=1)
    cam_x, cam_y = camera_mapped[:, 0], camera_mapped[:, 1]
    
    is_hidden = (cam_x < 0) | (cam_x >= frame_width) | (cam_y < 0) | (cam_y >= frame_height)
    pitch_map[is_hidden.reshape((canvas_height, canvas_width))] = (100, 100, 100) 
    
    cv2.rectangle(pitch_map, (0, 0), (canvas_width - 1, canvas_height - 1), (255, 255, 255), 2)
    
    mid_y = canvas_height // 2
    cv2.line(pitch_map, (0, mid_y), (canvas_width, mid_y), (255, 255, 255), 2)
    
    scale_factor = canvas_width // FIELD_WIDTH_M
    circle_radius = int(6 * scale_factor)
    cv2.circle(pitch_map, (canvas_width // 2, mid_y), circle_radius, (255, 255, 255), 2)
    
    return pitch_map
    

def run_interactive_calibration(frame):
    """
    Handles manual clicking to extrapolate the 4 pitch corners and sample
    anchor jersey colors from the environment.
    """
    pts = []
    img_copy = frame.copy()
    
    # Pre-calculate HSV image for color extraction
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            
            # Change color markers based on selection phase
            if len(pts) <= 8:
                color = (0, 0, 255) # Red for lines
                label = str(len(pts))
            elif len(pts) == 9:
                color = (255, 0, 0) # Blue tag for Team Blue
                label = "BLUE ANCHOR"
            else:
                color = (0, 255, 255) # Yellow tag for Team Yellow
                label = "YELLOW ANCHOR"
                
            cv2.circle(img_copy, (x, y), 5, color, -1)
            cv2.putText(img_copy, label, (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow("Calibration Phase", img_copy)

    cv2.namedWindow("Calibration Phase")
    cv2.setMouseCallback("Calibration Phase", click_event)
    
    print("\n[!] FOLLOW THIS EXACT ORDER FOR CLICKING:")
    print("1 & 2: LEFT Sideline (Click top/far point first, then bottom/near point)")
    print("3 & 4: RIGHT Sideline (Click top/far point first, then bottom/near point)")
    print("5 & 6: TOP/FAR Goal Line (Click left point first, then right point)")
    print("7 & 8: BOTTOM/NEAR Goal Line (Click left point first, then right point)")
    print("9:     Click directly on a Team1 jersey/vest")
    print("10:    Click directly on a Team2 jersey/vest")
    print("Press 'q' once all 10 points are clicked.")
    
    while len(pts) < 10:
        cv2.imshow("Calibration Phase", img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    if len(pts) < 10:
        raise ValueError("Calibration aborted. All 10 configuration points are required.")

    # Process field geometry lines
    left_sideline  = (pts[0], pts[1])
    right_sideline = (pts[2], pts[3])
    top_goal_line  = (pts[4], pts[5])
    bottom_goal_line = (pts[6], pts[7])

    c_tl = find_intersection(left_sideline[0], left_sideline[1], top_goal_line[0], top_goal_line[1])
    c_tr = find_intersection(right_sideline[0], right_sideline[1], top_goal_line[0], top_goal_line[1])
    c_br = find_intersection(right_sideline[0], right_sideline[1], bottom_goal_line[0], bottom_goal_line[1])
    c_bl = find_intersection(left_sideline[0], left_sideline[1], bottom_goal_line[0], bottom_goal_line[1])
    computed_corners = np.array([c_tl, c_tr, c_br, c_bl], dtype=np.float32)
    
    # Extract exact HSV anchor values from pixels clicked at steps 9 and 10
    team1_pixel_hsv = hsv_frame[pts[8][1], pts[8][0]].tolist()
    team2_pixel_hsv = hsv_frame[pts[9][1], pts[9][0]].tolist()
    
    print(f"[+] Sampled Team1 Jersey HSV Anchor: {team1_pixel_hsv}")
    print(f"[+] Sampled Team2 Jersey HSV Anchor: {team2_pixel_hsv}")
    
    return computed_corners, team1_pixel_hsv, team2_pixel_hsv

def get_homography_matrix(video_path):
    """Loads existing calibration matrix or builds a new one dynamically."""
    config_path = "camera_calibration.json"
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    scale_factor, canvas_width, canvas_height = calculate_dynamic_dimensions(frame_width, frame_height)
    
    if os.path.exists(config_path):
        print("[+] Loading saved camera calibration file...")
        with open(config_path, "r") as f:
            data = json.load(f)
            
        # Ensure older file formats update gracefully if anchor data is missing
        if "team1_anchor_hsv" not in data:
            print("[!] Warning: Saved configuration lacks jersey color anchors. Deleting configuration to force re-calibration...")
            cap.release()
            os.remove(config_path)
            return get_homography_matrix(video_path)
            
        return np.array(data["matrix"], dtype=np.float32), scale_factor, canvas_width, canvas_height
    
    print("[+] Configuration not found. Initiating calibration...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read video frame for calibration.")
        
    corners, team1_hsv, team2_hsv = run_interactive_calibration(frame)
    
    dst_points = np.array([
        [0, 0],                         
        [canvas_width, 0],               
        [canvas_width, canvas_height],    
        [0, canvas_height]               
    ], dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(corners, dst_points)
        
    # Compile and save matrix along with adaptive color profile targets
    calibration_payload = {
        "matrix": H.tolist(),
        "scale_factor": scale_factor,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "team1_anchor_hsv": team1_hsv,
        "team2_anchor_hsv": team2_hsv
    }
    
    with open(config_path, "w") as f:
        json.dump(calibration_payload, f, indent=4)
    print(f"[+] Calibration configuration successfully saved to {config_path}")
    
    blind_spot_map = generate_blind_spot_map(H, canvas_width, canvas_height, frame_width, frame_height)
    
    print("\n[+] Displaying Field Visibility Map. Hidden zones are shaded GREY.")
    print("Press any key on the image window to close it and start video analytics.")
    
    cv2.imshow("Field Visibility & Blind Spots", blind_spot_map)
    cv2.imwrite("field_visibility_map.jpg", blind_spot_map) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return H, scale_factor, canvas_width, canvas_height

# ==========================================
# MODULE 2: TEAM CLASSIFICATION (JERSEY COLOR)
# ==========================================
def assign_teams(frame, detections):
    if len(detections) < 4:
        return [0] * len(detections)
        
    colors = []
    for bbox in detections.xyxy:
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        torso = frame[int(y1 + height*0.2):int(y1 + height*0.5), x1:x2]
        
        if torso.size == 0:
            colors.append([0, 0, 0])
            continue
            
        average_color = torso.mean(axis=0).mean(axis=0) 
        colors.append(average_color)
        
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(colors)
    return kmeans.labels_


# ==========================================
# MODULE 3: METRIC TRANSFORMATIONS
# ==========================================
def get_metric_coordinates(x_pixel, y_pixel, H):
    """Transforms standard frame coordinates directly into real-world meters."""
    point = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    
    cx, cy = transformed[0][0][0], transformed[0][0][1]
    
    # Fallback placeholder scale initialization to maintain back-compatibility 
    # if global SCALE_FACTOR scope varies outside module injection
    current_scale = 20
    
    mx = cx / current_scale
    my = cy / current_scale
    return mx, my