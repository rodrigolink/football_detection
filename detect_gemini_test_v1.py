import random
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans

# Import calibration helpers from your original file
from detect_gemini_v1 import get_homography_matrix, get_metric_coordinates

def sample_torso_pixels(frame, bbox):
    """
    Crops the player's torso area to sample their actual vest color.
    Returns the average BGR color of that region.
    """
    x1, y1, x2, y2 = map(int, bbox)
    w = x2 - x1
    h = y2 - y1
    
    # Target the middle section of their upper body to avoid shorts/shoes/grass
    torso = frame[int(y1 + h * 0.2):int(y1 + h * 0.5), int(x1 + w * 0.15):int(x2 - w * 0.15)]
    
    if torso.size == 0:
        return np.array([128, 128, 128], dtype=np.float32)
    
    return torso.mean(axis=0).mean(axis=0)

def generate_tactical_birds_eye(players_list, canvas_width, canvas_height, goalie_a_idx, goalie_b_idx):
    """Generates a cushioned 2D top-down radar view including out-of-bounds players."""
    # Create an slightly expanded black/green base canvas to show buffer zones clearly
    pitch_map = np.zeros((canvas_height + 40, canvas_width + 40, 3), dtype=np.uint8)
    pitch_map[:] = (25, 80, 25) # Darker green background for the out-of-bounds area
    
    # Internal baseline offsets so lines are shifted inward from the image border
    offset_x = 20
    offset_y = 20
    
    # Draw true playable field lines (the white box)
    cv2.rectangle(pitch_map, (offset_x, offset_y), (canvas_width + offset_x, canvas_height + offset_y), (255, 255, 255), 2)
    mid_y = (canvas_height // 2) + offset_y
    cv2.line(pitch_map, (offset_x, mid_y), (canvas_width + offset_x, mid_y), (255, 255, 255), 2)
    
    scale_factor = canvas_width // 25
    cv2.circle(pitch_map, (canvas_width // 2 + offset_x, mid_y), int(6 * scale_factor), (255, 255, 255), 2)

    COLOR_BLUE   = (255, 50, 50)
    COLOR_YELLOW = (0, 240, 240)
    COLOR_PINK   = (255, 0, 255)
    COLOR_GREEN  = (0, 255, 0)
    
    for p in players_list:
        # Calculate canvas pixel coordinates, including their cushion offset displacements
        cx_canvas = int(p["mx"] * scale_factor) + offset_x
        cy_canvas = int(p["my"] * scale_factor) + offset_y
        
        if p["index"] == goalie_a_idx:
            dot_color = COLOR_PINK
        elif p["index"] == goalie_b_idx:
            dot_color = COLOR_GREEN
        elif p["visual_team_cluster"] == 0:
            dot_color = COLOR_BLUE
        else:
            dot_color = COLOR_YELLOW
            
        cv2.circle(pitch_map, (cx_canvas, cy_canvas), 8, dot_color, -1)
        cv2.circle(pitch_map, (cx_canvas, cy_canvas), 8, (255, 255, 255), 1)
        cv2.putText(pitch_map, str(p["index"]), (cx_canvas - 5, cy_canvas + 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    return pitch_map

# ==============================================================================
# TWEAKED CONFIGURATION CONTROL PANEL
# ==============================================================================
YOLO_MODEL_WEIGHTS = "yolov8m.pt"

# Fix 1: Base thresholds to catch shadowed/far objects
PLAYER_CONF_BASE = 0.20   # Dropped slightly to catch far-field silhouettes
BALL_CONFIDENCE   = 0.15

# Fix 2: Tweak NMS Intersection-over-Union (IoU) 
# Raising this threshold prevents YOLO from combining overlapping players into one box
PLAYER_IOU_THRESHOLD = 0.65 

FIELD_WIDTH_M  = 25.0
FIELD_HEIGHT_M = 45.0
FIELD_CUSHION_METERS = 2.0

X_MIN_BOUND = 0.0 - FIELD_CUSHION_METERS
X_MAX_BOUND = FIELD_WIDTH_M + FIELD_CUSHION_METERS
Y_MIN_BOUND = 0.0 - FIELD_CUSHION_METERS
Y_MAX_BOUND = FIELD_HEIGHT_M + FIELD_CUSHION_METERS


def inspect_random_match_frame(video_path):
    import detect_gemini_v1
    H, scale_factor, canvas_width, canvas_height = get_homography_matrix(video_path)
    detect_gemini_v1.SCALE_FACTOR = scale_factor
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_target_frame = random.randint(int(total_frames * 0.25), int(total_frames * 0.75))
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[-] Frame extraction failed.")
        return

    # 3. Enhanced Inference Model Execution
    model = YOLO(YOLO_MODEL_WEIGHTS)
    
    # We explicitly declare iou=PLAYER_IOU_THRESHOLD here to resolve the near-left player merging issue!
    results = model(frame, verbose=False, iou=PLAYER_IOU_THRESHOLD)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    valid_classes = (detections.class_id == 0) | (detections.class_id == 32)
    detections = detections[valid_classes]
    
    # 4. Filter and Transform with Dynamic Distant-Zone Boosting
    all_players = []
    ball_bbox = None
    
    for idx, (bbox, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = y2 if class_id == 0 else (y1 + y2) / 2.0
        
        # Geometrical conversion to check distance from camera
        mx, my = get_metric_coordinates(cx, cy, H)
        
        if not (X_MIN_BOUND <= mx <= X_MAX_BOUND and Y_MIN_BOUND <= my <= Y_MAX_BOUND):
            continue
            
        if class_id == 32:
            if confidence >= BALL_CONFIDENCE:
                ball_bbox = bbox
            continue
            
        # --- FIX 1: DYNAMIC FAR-ZONE CONFIDENCE BOOSTING ---
        # The further away the player is (closer to Y=0), the more leniency we grant them.
        # This keeps near spectators blocked but extracts the far goalie from the dark shadows.
        dynamic_threshold = PLAYER_CONF_BASE
        if my < 15.0:  # If the player is in the top 1/3 section of the pitch
            dynamic_threshold = 0.12  # Aggressive reduction to catch shadowed silhouettes
            
        if confidence < dynamic_threshold:
            continue  # Drop noise that fails our dynamic scale map
            
        all_players.append({
            "index": idx,
            "bbox": bbox,
            "mx": mx,
            "my": my,
            "torso_bgr": sample_torso_pixels(frame, bbox)
        })

    # 5. Extract Goalies using proximity targets
    goal_a_target = np.array([12.5, 0.0])
    goal_b_target = np.array([12.5, 45.0])
    
    if len(all_players) >= 2:
        goalie_a_idx = min(all_players, key=lambda p: np.linalg.norm(np.array([p["mx"], p["my"]]) - goal_a_target))["index"]
        goalie_b_idx = min(all_players, key=lambda p: np.linalg.norm(np.array([p["mx"], p["my"]]) - goal_b_target))["index"]
    else:
        print("[-] Warning: Missing critical player counts for game setup calculation.")
        return


    # 5. Extract the Goalies (Anchored perfectly to the TRUE goal line targets)
    goal_a_target = np.array([12.5, 0.0])   # Fixed center of Top Goal
    goal_b_target = np.array([12.5, 45.0])  # Fixed center of Bottom Goal
    
    goalie_a_idx = min(all_players, key=lambda p: np.linalg.norm(np.array([p["mx"], p["my"]]) - goal_a_target))["index"]
    goalie_b_idx = min(all_players, key=lambda p: np.linalg.norm(np.array([p["mx"], p["my"]]) - goal_b_target))["index"]
    
    # 6. --- VISUAL JERSEY CLUSTERING (THE KEY FIX) ---
    # Isolate outfield player colors to segment the two teams visually
    outfield_players = [p for p in all_players if p["index"] not in (goalie_a_idx, goalie_b_idx)]
    
    if len(outfield_players) >= 2:
        color_features = [p["torso_bgr"] for p in outfield_players]
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(color_features)
        
        # Save the true color cluster values (0 or 1) back into our profiles
        for p, label in zip(outfield_players, kmeans.labels_):
            p["visual_team_cluster"] = label
    else:
        # Fallback if no outfield players are around
        for p in outfield_players:
            p["visual_team_cluster"] = 0

    # Explicitly set dummy variables for goalies so the drawing loop passes smoothly
    for p in all_players:
        if p["index"] == goalie_a_idx or p["index"] == goalie_b_idx:
            p["visual_team_cluster"] = -1

    # 7. Render Bounding Boxes Using True Visual Identity
    display_frame = frame.copy()
    COLOR_BLUE   = (255, 50, 50)
    COLOR_YELLOW = (0, 240, 240)
    COLOR_PINK   = (255, 0, 255)
    COLOR_GREEN  = (0, 255, 0)
    COLOR_ORANGE = (0, 140, 255)

    if ball_bbox is not None:
        x1, y1, x2, y2 = map(int, ball_bbox)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLOR_ORANGE, 2)

    for p in all_players:
        x1, y1, x2, y2 = map(int, p["bbox"])
        idx = p["index"]
        mx, my = p["mx"], p["my"]
        
        if idx == goalie_a_idx:
            box_color = COLOR_PINK
            label = f"Goalie Top ({mx:.1f}m, {my:.1f}m)"
        elif idx == goalie_b_idx:
            box_color = COLOR_GREEN
            label = f"Goalie Bottom ({mx:.1f}m, {my:.1f}m)"
        elif p["visual_team_cluster"] == 0:
            box_color = COLOR_BLUE
            label = f"Team Blue ({mx:.1f}m, {my:.1f}m)"
        else:
            box_color = COLOR_YELLOW
            label = f"Team Yellow ({mx:.1f}m, {my:.1f}m)"
            
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 2)

    # 8. Generate Birds Eye Graph
    birds_eye_map = generate_tactical_birds_eye(all_players, canvas_width, canvas_height, goalie_a_idx, goalie_b_idx)
    
    cv2.imshow("1. Verified Visual Color Tracking", display_frame)
    cv2.imshow("2. Tactical Radar Layout Map", birds_eye_map)
    cv2.imwrite("output_camera_view.jpg", display_frame)
    cv2.imwrite("output_birds_eye_view.jpg", birds_eye_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Put your actual video file name here!
    my_video = "game.mp4" 
    
    # Run the validation function we just built
    inspect_random_match_frame(my_video)