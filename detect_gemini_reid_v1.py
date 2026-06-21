import random
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import sys
import json

# ==============================================================================
# SECTION 1: TWEAKABLE CONFIGURATION CONTROL PANEL
# ==============================================================================
YOLO_MODEL_WEIGHTS = "yolo11m.pt" 
TRACKING_METHOD    = "botsort.yaml"  # Upgraded: Uses ReID embeddings + Camera Motion Compensation

TRACKING_WINDOW_FRAMES = 300  # 30 seconds at 30fps

# Object Detection Sensitivity
PLAYER_CONF_BASE     = 0.10   
PLAYER_IOU_THRESHOLD  = 0.60 

# Pitch Proportions (Real-World Meters)
FIELD_WIDTH_M        = 25.0
FIELD_HEIGHT_M       = 45.0

# --- THE CUSHION RETURN ---
# Tracks players up to this many meters outside the lines for throw-ins/corners
FIELD_CUSHION_METERS = 2.0  

# Smoothing Control (Exponential Moving Average Filter)
TRACK_SMOOTHING_ALPHA = 0.18  

# Export File Names
OUTPUT_CAMERA_VIDEO = "export_camera_tracked.mp4"
OUTPUT_RADAR_VIDEO  = "export_tactical_radar.mp4"

# ==============================================================================
# SECTION 2: UTILITY IMAGE PROCESSING FUNCTIONS
# ==============================================================================
def print_progress(current, total, phase_name="Processing"):
    """Generates a synchronized terminal progress block."""
    p = current / total
    bar = "█" * int(40 * p) + "-" * (40 - int(40 * p))
    sys.stdout.write(f"\r{phase_name} |{bar}| {p*100:.1f}% ({current}/{total})")
    sys.stdout.flush()


def classify_torso_by_anchors(frame, bbox, team1_bounds, team2_bounds):
    """
    Counts matching pixels inside the upper torso using dynamically constructed
    HSV masks generated from the custom calibrated anchor points.
    """
    x1, y1, x2, y2 = map(int, bbox)
    w = x2 - x1
    h = y2 - y1
    
    torso_top = max(0, y1 + int(h * 0.15))
    torso_bottom = max(0, y1 + int(h * 0.45))
    sliver_left = max(0, x1 + int(w * 0.25))
    sliver_right = max(0, x2 - int(w * 0.25))
    
    torso_sliver = frame[torso_top:torso_bottom, sliver_left:sliver_right]
    if torso_sliver.size == 0:
        return None
        
    hsv = cv2.cvtColor(torso_sliver, cv2.COLOR_BGR2HSV)
    
    lower_t1, upper_t1 = team1_bounds
    lower_t2, upper_t2 = team2_bounds
    
    mask_t1 = cv2.inRange(hsv, lower_t1, upper_t1)
    mask_t2 = cv2.inRange(hsv, lower_t2, upper_t2)
    
    count_t1 = cv2.countNonZero(mask_t1)
    count_t2 = cv2.countNonZero(mask_t2)
    
    MIN_CONFIDENCE_PIXELS = 5 
    
    if count_t1 > count_t2 and count_t1 > MIN_CONFIDENCE_PIXELS:
        return 0  
    elif count_t2 > count_t1 and count_t2 > MIN_CONFIDENCE_PIXELS:
        return 1  
        
    return None


def get_metric_coordinates_local(x_pixel, y_pixel, H, scale_factor):
    """Transforms foot pixel directly to meters on the radar view."""
    point = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    cx, cy = transformed[0][0][0], transformed[0][0][1]
    return cx / scale_factor, cy / scale_factor


def generate_tactical_birds_eye(active_players, canvas_width, canvas_height, goalie_top_set, goalie_bottom_set, team_assignments):
    """Generates the 2D top-down tactical layout board."""
    pitch_map = np.zeros((canvas_height + 40, canvas_width + 40, 3), dtype=np.uint8)
    pitch_map[:] = (40, 110, 40) 
    
    offset = 20
    cv2.rectangle(pitch_map, (offset, offset), (canvas_width + offset, canvas_height + offset), (255, 255, 255), 2)
    mid_y = (canvas_height // 2) + offset
    cv2.line(pitch_map, (offset, mid_y), (canvas_width + offset, mid_y), (255, 255, 255), 2)
    
    scale_factor = canvas_width // int(FIELD_WIDTH_M)
    cv2.circle(pitch_map, (canvas_width // 2 + offset, mid_y), int(6 * scale_factor), (255, 255, 255), 2)

    for p in active_players:
        pid = p["id"]
        cx = int(p["mx"] * scale_factor) + offset
        cy = int(p["my"] * scale_factor) + offset
        
        if pid in goalie_top_set:
            color = (255, 0, 255) 
        elif pid in goalie_bottom_set:
            color = (0, 255, 0)   
        elif team_assignments.get(pid, 0) == 0:
            color = (255, 50, 50) 
        else:
            color = (0, 240, 240) 
            
        cv2.circle(pitch_map, (cx, cy), 8, color, -1)
        cv2.circle(pitch_map, (cx, cy), 8, (255, 255, 255), 1)
        cv2.putText(pitch_map, str(pid), (cx - 5, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    return pitch_map


# ==============================================================================
# SECTION 3: CORE SINGLE-PASS PROCESSING & COMPILATION ENGINE
# ==============================================================================
def process_and_export_30s_tracking(video_path):
    from detect_gemini_v1 import get_homography_matrix
    H, scale_factor, canvas_width, canvas_height = get_homography_matrix(video_path)
    
    with open("camera_calibration.json", 'r') as f:
        calib_data = json.load(f)
        
    anchor_t1 = calib_data["team1_anchor_hsv"]
    anchor_t2 = calib_data["team2_anchor_hsv"]
    
    # Adaptive HSV limits
    lower_t1 = np.array([max(0, anchor_t1[0] - 12), 65, 55])
    upper_t1 = np.array([min(179, anchor_t1[0] + 12), 255, 255])
    team1_bounds = (lower_t1, upper_t1)
    
    lower_t2 = np.array([max(0, anchor_t2[0] - 12), 65, 55])
    upper_t2 = np.array([min(179, anchor_t2[0] + 12), 255, 255])
    team2_bounds = (lower_t2, upper_t2)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
# Calculate the boundaries (between 25% and 75% of the video)
    min_start_frame = int(total_frames * 0.25)
    max_start_frame = int(total_frames * 0.75)
    
    if total_frames <= TRACKING_WINDOW_FRAMES:
        start_frame = 0
        actual_processing_frames = total_frames
    else:
        # Pick a completely random starting frame strictly within that middle 50% window
        start_frame = random.randint(min_start_frame, max_start_frame)
        
        # Guard rail: Ensure we don't accidentally overflow past the end of the video
        if start_frame + TRACKING_WINDOW_FRAMES > total_frames:
            start_frame = total_frames - TRACKING_WINDOW_FRAMES - 1
            
        actual_processing_frames = TRACKING_WINDOW_FRAMES
        
    model = YOLO(YOLO_MODEL_WEIGHTS)
    
    # Persistent Registers
    history_colors = {}
    history_tracks_exist = set()  
    smoothed_positions = {}
    top_goalie_votes = {}
    bottom_goalie_votes = {}
    
    frame_buffer = []
    tracking_data_buffer = []
    
    GOAL_TOP_COORD = np.array([12.5, 0.0])
    GOAL_BOTTOM_COORD = np.array([12.5, 45.0])
    
    # --- DYNAMIC CUSHION POLYGON GENERATION ---
    cushion_px = int(FIELD_CUSHION_METERS * scale_factor)
    padded_polygon = np.array([
        [-cushion_px, -cushion_px],
        [canvas_width + cushion_px, -cushion_px],
        [canvas_width + cushion_px, canvas_height + cushion_px],
        [-cushion_px, canvas_height + cushion_px]
    ], dtype=np.int32)
    
    # --------------------------------------------------------------------------
    # PASS 1: TRACK ONCE & RECORD TRAJECTORIES (Upgraded with Embedded ReID)
    # --------------------------------------------------------------------------
    print("[+] Executing Phase 1/2: High-Resolution Deep ReID Track Analytics")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for f_idx in range(actual_processing_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        big_frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # --- THE REID UPGRADE ---
        # Leverages YOLO's native embedding-driven tracking loop directly.
        results = model.track(
            big_frame, 
            imgsz=1920, 
            conf=PLAYER_CONF_BASE, 
            iou=PLAYER_IOU_THRESHOLD,
            persist=True,               # Vital: Forces memory persistence across frames
            tracker=TRACKING_METHOD,     # Loads BoT-SORT configuration
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        
        # Enforce that objects must be 'person' class AND have a valid, assigned tracker ID
        player_detections = detections[
            (detections.class_id == 0) & (detections.tracker_id != None)
        ]
        
        if len(player_detections.xyxy) > 0:
            player_detections.xyxy //= 2
            
        current_frame_players = []
        
        for bbox, _, _, _, tracker_id, _ in player_detections:
            # Cast tracker_id to standard int for downstream compatibility
            tracker_id = int(tracker_id)
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2.0, y2
            
            point = np.array([[[cx, cy]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, H)[0][0]
            canvas_x, canvas_y = transformed[0], transformed[1]
            
            is_inside = cv2.pointPolygonTest(padded_polygon, (float(canvas_x), float(canvas_y)), False)
            if is_inside < 0:
                continue 
                
            raw_mx = canvas_x / scale_factor
            raw_my = canvas_y / scale_factor
            
            # EMA Position Smoothing
            if tracker_id not in smoothed_positions:
                smoothed_positions[tracker_id] = (raw_mx, raw_my)
            else:
                prev_mx, prev_my = smoothed_positions[tracker_id]
                alpha = TRACK_SMOOTHING_ALPHA
                smoothed_mx = (alpha * raw_mx) + ((1.0 - alpha) * prev_mx)
                smoothed_my = (alpha * raw_my) + ((1.0 - alpha) * prev_my)
                smoothed_positions[tracker_id] = (smoothed_mx, smoothed_my)
            
            mx, my = smoothed_positions[tracker_id]
            history_tracks_exist.add(tracker_id) 
            
            current_frame_players.append({
                "id": tracker_id, "bbox": bbox, "mx": mx, "my": my
            })
            
            assigned_vote = classify_torso_by_anchors(frame, bbox, team1_bounds, team2_bounds)
            if assigned_vote is not None:
                if tracker_id not in history_colors:
                    history_colors[tracker_id] = []
                history_colors[tracker_id].append(assigned_vote)
                
        # Goalie Proximity Voting Maps
        if len(current_frame_players) >= 2:
            closest_top = min(current_frame_players, key=lambda p: np.linalg.norm(np.array([p["mx"], p["my"]]) - GOAL_TOP_COORD))
            closest_bottom = min(current_frame_players, key=lambda p: np.linalg.norm(np.array([p["mx"], p["my"]]) - GOAL_BOTTOM_COORD))
            
            if np.linalg.norm(np.array([closest_top["mx"], closest_top["my"]]) - GOAL_TOP_COORD) < 6.0:
                top_goalie_votes[closest_top["id"]] = top_goalie_votes.get(closest_top["id"], 0) + 1
            if np.linalg.norm(np.array([closest_bottom["mx"], closest_bottom["my"]]) - GOAL_BOTTOM_COORD) < 6.0:
                bottom_goalie_votes[closest_bottom["id"]] = bottom_goalie_votes.get(closest_bottom["id"], 0) + 1

        frame_buffer.append(frame.copy())
        tracking_data_buffer.append(current_frame_players)
        print_progress(f_idx + 1, actual_processing_frames, phase_name="Pass 1/2 (Inference)")

    cap.release()
    print("\n") 

    # --------------------------------------------------------------------------
    # POST-PROCESSING: RESOLVE TEAMS VIA LIFETIME HISTORICAL VOTES
    # --------------------------------------------------------------------------
    all_tracked_ids = list(history_tracks_exist)
    if len(all_tracked_ids) < 2:
        print("[-] Error: Insufficient player tracks detected to compile video maps.")
        return
        
    confirmed_top_goalie_ids = {pid for pid, votes in top_goalie_votes.items() if votes > 15}
    confirmed_bottom_goalie_ids = {pid for pid, votes in bottom_goalie_votes.items() if votes > 15}
    all_goalie_ids = confirmed_top_goalie_ids.union(confirmed_bottom_goalie_ids)
    
    outfield_ids = [pid for pid in all_tracked_ids if pid not in all_goalie_ids]
    team_assignments = {}
    
    for pid in outfield_ids:
        if pid in history_colors and len(history_colors[pid]) > 0:
            votes = history_colors[pid]
            team_assignments[pid] = 0 if votes.count(0) > votes.count(1) else 1
        else:
            team_assignments[pid] = 0

    # --------------------------------------------------------------------------
    # PASS 2: RENDER BUFFERED ARRAYS DIRECTLY TO DISK STREAMS
    # --------------------------------------------------------------------------
    print("[+] Executing Phase 2/2: Mapping Visual Classes & Map Video Compilations")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_camera = cv2.VideoWriter(OUTPUT_CAMERA_VIDEO, fourcc, fps, (frame_width, frame_height))
    
    radar_w, radar_h = canvas_width + 40, canvas_height + 40
    writer_radar = cv2.VideoWriter(OUTPUT_RADAR_VIDEO, fourcc, fps, (radar_w, radar_h))
    
    total_render_frames = len(frame_buffer)
    
    for r_idx, (raw_frame, frame_players) in enumerate(zip(frame_buffer, tracking_data_buffer)):
        display_frame = raw_frame.copy()
        
        for p in frame_players:
            pid = p["id"]
            x1, y1, x2, y2 = map(int, p["bbox"])
            
            if pid in confirmed_top_goalie_ids:
                color, label = (255, 0, 255), "Goalie Top"
            elif pid in confirmed_bottom_goalie_ids:
                color, label = (0, 255, 0), "Goalie Bottom"
            elif team_assignments.get(pid, 0) == 0:
                color, label = (50, 50, 255), "Team Color 1"
            else:
                color, label = (0, 240, 240), "Team Color 2"
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{label} [ID {pid}]", (x1, y1 - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            
        radar_frame = generate_tactical_birds_eye(frame_players, canvas_width, canvas_height, 
                                                confirmed_top_goalie_ids, confirmed_bottom_goalie_ids, team_assignments)
        
        writer_camera.write(display_frame)
        writer_radar.write(radar_frame)
        print_progress(r_idx + 1, total_render_frames, phase_name="Pass 2/2 (Rendering)")
        
    writer_camera.release()
    writer_radar.release()
    
    print(f"\n\n[+] Processing completed successfully!")


if __name__ == "__main__":
    my_video = "game.mp4" 
    process_and_export_30s_tracking(my_video)