import random
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import sys
import json
import logging

# Force Ultralytics loggers to silence background metrics.
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ==============================================================================
# SECTION 1: TWEAKABLE CONFIGURATION CONTROL PANEL
# ==============================================================================
YOLO_MODEL_WEIGHTS = "yolo11m.pt"   

# Frame Skipping Control
FRAME_SKIP_STEP = 6  

# Total structural frame window size to analyze (in absolute source frames)
TRACKING_WINDOW_FRAMES = 18000  

# DYNAMIC ENGINE CALIBRATION
PLAYER_CONF_BASE     = 0.12   # Dropped slightly to capture the fast-moving blue/yellow players
PLAYER_IOU_THRESHOLD  = 0.65   # Balanced to keep nearby players separated
BALL_CONFIDENCE       = 0.15  

# Pitch Proportions (Real-World Meters)
FIELD_WIDTH_M        = 25.0
FIELD_HEIGHT_M       = 45.0
FIELD_CUSHION_METERS = 6.0  # Safe boundary margin to prevent edge drops

# Smoothing Control
TRACK_SMOOTHING_ALPHA = 0.18  

# Export File Names
OUTPUT_CAMERA_VIDEO = "export_camera_tracked.mp4"
OUTPUT_RADAR_VIDEO  = "export_tactical_radar.mp4"

# ==============================================================================
# SECTION 2: UTILITY IMAGE PROCESSING FUNCTIONS
# ==============================================================================
def print_progress(current, total, phase_name="Processing"):
    """Generates a synchronized terminal progress block that forces line overwrites."""
    p = current / total if total > 0 else 0
    bar = "█" * int(33 * p) + "-" * (33 - int(33 * p))
    sys.stdout.write(f"\r\033[K{phase_name} |{bar}| {p*100:.1f}% ({current}/{total})")
    sys.stdout.flush()


def classify_torso_by_anchors(frame, bbox, team1_bounds, team2_bounds):
    """ULTRA-TIGHT CROPPING: Focuses purely on the innermost spine line of the player."""
    x1, y1, x2, y2 = map(int, bbox)
    w = x2 - x1
    h = y2 - y1
    
    # Advanced center mass isolation to eliminate color bleeding from nearby players
    torso_top = max(0, y1 + int(h * 0.18))
    torso_bottom = max(0, y1 + int(h * 0.42))
    sliver_left = max(0, x1 + int(w * 0.33))   # Shrunk from 0.25 to aggressively target the shirt center
    sliver_right = max(0, x2 - int(w * 0.33))
    
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
    
    MIN_CONFIDENCE_PIXELS = 6 
    
    if count_t1 > count_t2 and count_t1 > MIN_CONFIDENCE_PIXELS:
        return 0  
    elif count_t2 > count_t1 and count_t2 > MIN_CONFIDENCE_PIXELS:
        return 1  
    return None


def generate_tactical_birds_eye(active_players, ball_coords, canvas_width, canvas_height, goalie_top_set, goalie_bottom_set, team_assignments):
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

    if ball_coords is not None:
        bx = int(ball_coords[0] * scale_factor) + offset
        by = int(ball_coords[1] * scale_factor) + offset
        cv2.circle(pitch_map, (bx, by), 5, (255, 255, 255), -1)
        cv2.circle(pitch_map, (bx, by), 5, (0, 0, 0), 1)

    return pitch_map


# ==============================================================================
# SECTION 3: CORE ROTATION-AWARE PROCESSING ENGINE
# ==============================================================================
def process_and_export_30s_tracking(video_path):
    from detect_gemini_v1 import get_homography_matrix
    H, scale_factor, canvas_width, canvas_height = get_homography_matrix(video_path)
    
    with open("camera_calibration.json", 'r') as f:
        calib_data = json.load(f)
        
    anchor_t1 = calib_data["team1_anchor_hsv"]
    anchor_t2 = calib_data["team2_anchor_hsv"]
    
    # Balanced Hue ranges to manage high glare from stadium lights
    team1_bounds = (np.array([max(0, anchor_t1[0] - 14), 60, 50]), np.array([min(179, anchor_t1[0] + 14), 255, 255]))
    team2_bounds = (np.array([max(0, anchor_t2[0] - 14), 60, 50]), np.array([min(179, anchor_t2[0] + 14), 255, 255]))

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    min_start_frame = int(total_frames * 0.25)
    max_start_frame = int(total_frames * 0.75)
    
    if total_frames <= TRACKING_WINDOW_FRAMES:
        start_frame = 0
        actual_processing_steps = total_frames // FRAME_SKIP_STEP
    else:
        start_frame = random.randint(min_start_frame, max_start_frame)
        if start_frame + TRACKING_WINDOW_FRAMES > total_frames:
            start_frame = total_frames - TRACKING_WINDOW_FRAMES - 1
        actual_processing_steps = TRACKING_WINDOW_FRAMES // FRAME_SKIP_STEP
        
    model = YOLO(YOLO_MODEL_WEIGHTS).to("mps")
    
    history_colors = {}
    smoothed_positions = {}
    top_goalie_votes = {}
    bottom_goalie_votes = {}
    
    tracking_data_buffer = []
    ball_data_buffer = []  
    
    GOAL_TOP_COORD = np.array([12.5, 0.0])
    GOAL_BOTTOM_COORD = np.array([12.5, 45.0])
    
    cushion_px = int(FIELD_CUSHION_METERS * scale_factor)
    padded_polygon = np.array([
        [-cushion_px, -cushion_px],
        [canvas_width + cushion_px, -cushion_px],
        [canvas_width + cushion_px, canvas_height + cushion_px],
        [-cushion_px, canvas_height + cushion_px]
    ], dtype=np.int32)
    
    # --------------------------------------------------------------------------
    # PASS 1: METADATA STREAM TRACKING
    # --------------------------------------------------------------------------
    print(f"[+] Phase 1/2: GPU Accelerated Deep Tracking (MPS) | Step Stride: {FRAME_SKIP_STEP}")
    
    processed_count = 0
    fake_id_counter = 99000  # Placeholder IDs to assign to recovered un-tracked detections
    
    for f_idx in range(actual_processing_steps):
        target_frame = start_frame + (f_idx * FRAME_SKIP_STEP)
        if target_frame >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(
            frame, imgsz=1920, conf=PLAYER_CONF_BASE, iou=PLAYER_IOU_THRESHOLD,
            persist=True, verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        
        # RECOVERY STEP: Catch valid player boxes even if the internal tracking ID is missing
        player_detections = detections[(detections.class_id == 0)]
        ball_detections = detections[(detections.class_id == 32) & (detections.confidence >= BALL_CONFIDENCE)]

        current_frame_players = []
        
        for bbox, _, conf, _, tracker_id, _ in player_detections:
            # If tracking index dropped out during fast movement, assign a temporary recovery index
            if tracker_id is None:
                if conf > 0.40:  # High confidence check prevents turf textures from triggering
                    tracker_id = fake_id_counter
                    fake_id_counter += 1
                else:
                    continue
            else:
                tracker_id = int(tracker_id)
                
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2.0, y2
            
            point = np.array([[[cx, cy]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, H)[0][0]
            canvas_x, canvas_y = transformed[0], transformed[1]
            
            if cv2.pointPolygonTest(padded_polygon, (float(canvas_x), float(canvas_y)), False) < 0:
                continue 
                
            raw_mx, raw_my = canvas_x / scale_factor, canvas_y / scale_factor
            
            if tracker_id not in smoothed_positions:
                smoothed_positions[tracker_id] = (raw_mx, raw_my)
            else:
                prev_mx, prev_my = smoothed_positions[tracker_id]
                alpha = TRACK_SMOOTHING_ALPHA
                smoothed_positions[tracker_id] = ((alpha * raw_mx) + ((1.0 - alpha) * prev_mx),
                                                    (alpha * raw_my) + ((1.0 - alpha) * prev_my))
            
            mx, my = smoothed_positions[tracker_id]
            current_frame_players.append({"id": tracker_id, "bbox": bbox, "mx": mx, "my": my})
            
            assigned_vote = classify_torso_by_anchors(frame, bbox, team1_bounds, team2_bounds)
            if assigned_vote is not None:
                if tracker_id not in history_colors:
                    history_colors[tracker_id] = []
                history_colors[tracker_id].append(assigned_vote)

        current_frame_ball = None
        if len(ball_detections.xyxy) > 0:
            best_ball_idx = np.argmax(ball_detections.confidence)
            bx1, by1, bx2, by2 = ball_detections.xyxy[best_ball_idx]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0  
            
            b_point = np.array([[[bcx, bcy]]], dtype=np.float32)
            b_transformed = cv2.perspectiveTransform(b_point, H)[0][0]
            bmx, bmy = b_transformed[0] / scale_factor, b_transformed[1] / scale_factor
            
            if cv2.pointPolygonTest(padded_polygon, (float(b_transformed[0]), float(b_transformed[1])), False) >= 0:
                current_frame_ball = (bmx, bmy, (int(bx1), int(by1), int(bx2), int(by2)))
                
        if len(current_frame_players) >= 2:
            closest_top = min(current_frame_players, key=lambda p: np.linalg.norm(np.array([p["mx"], p["my"]]) - GOAL_TOP_COORD))
            closest_bottom = min(current_frame_players, key=lambda p: np.linalg.norm(np.array([p["mx"], p["my"]]) - GOAL_BOTTOM_COORD))
            
            if np.linalg.norm(np.array([closest_top["mx"], closest_top["my"]]) - GOAL_TOP_COORD) < 6.0:
                top_goalie_votes[closest_top["id"]] = top_goalie_votes.get(closest_top["id"], 0) + 1
            if np.linalg.norm(np.array([closest_bottom["mx"], closest_bottom["my"]]) - GOAL_BOTTOM_COORD) < 6.0:
                bottom_goalie_votes[closest_bottom["id"]] = bottom_goalie_votes.get(closest_bottom["id"], 0) + 1

        tracking_data_buffer.append(current_frame_players)
        ball_data_buffer.append(current_frame_ball)  
        processed_count += 1
        print_progress(processed_count, actual_processing_steps, phase_name="Pass 1/2 (Inference)")

    cap.release()
    print("\n") 

    # --------------------------------------------------------------------------
    # POST-PROCESSING: RESOLVE TEAMS USING TIMELINE-WIDE VOTING WEIGHTS
    # --------------------------------------------------------------------------
    confirmed_top_goalie_ids = {pid for pid, votes in top_goalie_votes.items() if votes > 15}
    confirmed_bottom_goalie_ids = {pid for pid, votes in bottom_goalie_votes.items() if votes > 15}
    all_goalie_ids = confirmed_top_goalie_ids.union(confirmed_bottom_goalie_ids)
    
    team_assignments = {}
    for pid, votes in history_colors.items():
        if pid in all_goalie_ids:
            continue
        if len(votes) > 0:
            team_assignments[pid] = 0 if votes.count(0) > votes.count(1) else 1

    # --------------------------------------------------------------------------
    # PASS 2: STREAM RE-OPEN & METRIC RENDERING
    # --------------------------------------------------------------------------
    print("[+] Phase 2/2: Streaming Video Rendering to Disk")
    cap_render = cv2.VideoCapture(video_path)
    
    output_fps = float(fps) / FRAME_SKIP_STEP
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer_camera = cv2.VideoWriter(OUTPUT_CAMERA_VIDEO, fourcc, output_fps, (frame_width, frame_height))
    writer_radar = cv2.VideoWriter(OUTPUT_RADAR_VIDEO, fourcc, output_fps, (canvas_width + 40, canvas_height + 40))
    
    for r_idx in range(processed_count):
        target_frame = start_frame + (r_idx * FRAME_SKIP_STEP)
        cap_render.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        ret, raw_frame = cap_render.read()
        if not ret:
            break
            
        frame_players = tracking_data_buffer[r_idx]
        frame_ball = ball_data_buffer[r_idx]
        display_frame = raw_frame.copy()
        
        for p in frame_players:
            pid = p["id"]
            x1, y1, x2, y2 = map(int, p["bbox"])
            
            current_frame_team = classify_torso_by_anchors(raw_frame, p["bbox"], team1_bounds, team2_bounds)
            if current_frame_team is not None and pid not in all_goalie_ids:
                team_assignments[pid] = current_frame_team
                
            assigned_team = team_assignments.get(pid, None)
            
            if pid in confirmed_top_goalie_ids:
                color, label = (255, 0, 255), "Goalie Top"
            elif pid in confirmed_bottom_goalie_ids:
                color, label = (0, 255, 0), "Goalie Bottom"
            elif assigned_team == 0:
                color, label = (50, 50, 255), "Team Color 1"
            elif assigned_team == 1:
                color, label = (0, 240, 240), "Team Color 2"
            else:
                color, label = (200, 200, 200), "Unknown Team"
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{label} [ID {pid}]", (x1, y1 - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            
        radar_ball_coords = None
        if frame_ball is not None:
            bmx, bmy, (bx1, by1, bx2, by2) = frame_ball
            radar_ball_coords = (bmx, bmy)
            cv2.circle(display_frame, ((bx1 + bx2) // 2, (by1 + by2) // 2), 5, (0, 140, 255), 2)
            
        radar_frame = generate_tactical_birds_eye(frame_players, radar_ball_coords, canvas_width, canvas_height, 
                                                confirmed_top_goalie_ids, confirmed_bottom_goalie_ids, team_assignments)
        
        writer_camera.write(display_frame)
        writer_radar.write(radar_frame)
        print_progress(r_idx + 1, processed_count, phase_name="Pass 2/2 (Rendering)")
        
    cap_render.release()
    writer_camera.release()
    writer_radar.release()
    print(f"\n\n[+] Processing completed successfully!")


if __name__ == "__main__":
    my_video = "game.mp4" 
    process_and_export_30s_tracking(my_video)