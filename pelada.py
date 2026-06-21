import random
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import sys
import json
import logging
import os
import time  # NEW: For measuring performance benchmarks

# Silence background training/metrics output from YOLO
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ==============================================================================
# SECTION 1: CONFIGURATION CONTROL PANEL
# ==============================================================================
YOLO_MODEL_WEIGHTS = "yolo11m.pt"   
PLAYER_CONF_BASE = 0.20  

FRAME_SKIP_STEP = 5  

# Snippet Window Size (In absolute video frames)
# Set to 0 to automatically bypass random slicing and process the entire video!
TRACKING_WINDOW_FRAMES = 1800  

# Pitch Proportions (Real-World Meters)
FIELD_WIDTH_M        = 25.0
FIELD_HEIGHT_M       = 45.0
FIELD_CUSHION_METERS = 3.0  

# JITTER & COLOR MITIGATION
TRACK_SMOOTHING_ALPHA = 0.22  
COLOR_SMOOTHING_ALPHA = 0.25  
MAX_MATCH_DISTANCE_M  = 3.5   

# Transition Detection Parameters
WINDOW_SECONDS = 5       
VARIANCE_THRESHOLD = 1.5 
COOLDOWN_SECONDS = 15    

# Export File Names
OUTPUT_CAMERA_VIDEO = "camera_tracked.mp4" 
OUTPUT_RADAR_VIDEO  = "pitch_view.mp4"
OUTPUT_JSON_LOG     = "match_transitions.json"

# ==============================================================================
# SECTION 2: UTILITY IMAGE PROCESSING FUNCTIONS
# ==============================================================================
def print_progress(current, total, phase_name="Processing"):
    p = current / total if total > 0 else 0
    bar = "█" * int(30 * p) + "-" * (30 - int(30 * p))
    sys.stdout.write(f"\r\033[K{phase_name} |{bar}| {p*100:.1f}% ({current}/{total})")
    sys.stdout.flush()

def frames_to_timestamp(frame_idx, fps):
    total_seconds = int(frame_idx / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def hsv_to_bgr(hsv_array):
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0] = hsv_array
    bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def classify_torso_color(frame, bbox, team1_bounds, team2_bounds):
    x1, y1, x2, y2 = map(int, bbox)
    w = x2 - x1
    h = y2 - y1
    
    torso_top = max(0, y1 + int(h * 0.18))
    torso_bottom = max(0, y1 + int(h * 0.42))
    sliver_left = max(0, x1 + int(w * 0.33))   
    sliver_right = max(0, x2 - int(w * 0.33))
    
    torso_sliver = frame[torso_top:torso_bottom, sliver_left:sliver_right]
    if torso_sliver.size == 0:
        return 0.5
        
    hsv = cv2.cvtColor(torso_sliver, cv2.COLOR_BGR2HSV)
    
    lower_t1, upper_t1 = team1_bounds
    lower_t2, upper_t2 = team2_bounds
    
    mask_t1 = cv2.inRange(hsv, lower_t1, upper_t1)
    mask_t2 = cv2.inRange(hsv, lower_t2, upper_t2)
    
    count_t1 = cv2.countNonZero(mask_t1)
    count_t2 = cv2.countNonZero(mask_t2)
    
    total_pixels = count_t1 + count_t2
    if total_pixels == 0:
        return 0.5
        
    return float(count_t2) / float(total_pixels)

def get_dynamic_team_color(score, color_t1, color_t2):
    if score < 0.42:
        return color_t1   
    elif score > 0.58:
        return color_t2   
    else:
        return (240, 240, 240)  

def generate_birds_eye(active_players, canvas_width, canvas_height, color_t1, color_t2):
    pitch_map = np.zeros((canvas_height + 40, canvas_width + 40, 3), dtype=np.uint8)
    pitch_map[:] = (40, 110, 40) 
    
    offset = 20
    cv2.rectangle(pitch_map, (offset, offset), (canvas_width + offset, canvas_height + offset), (255, 255, 255), 2)
    mid_y = (canvas_height // 2) + offset
    cv2.line(pitch_map, (offset, mid_y), (canvas_width + offset, mid_y), (255, 255, 255), 2)
    
    scale_factor = canvas_width // int(FIELD_WIDTH_M)
    cv2.circle(pitch_map, (canvas_width // 2 + offset, mid_y), int(6 * scale_factor), (255, 255, 255), 2)

    for p in active_players:
        cx = int(p["mx"] * scale_factor) + offset
        cy = int(p["my"] * scale_factor) + offset
        
        dot_color = get_dynamic_team_color(p["team_score"], color_t1, color_t2)
        
        cv2.circle(pitch_map, (cx, cy), 8, dot_color, -1)
        cv2.circle(pitch_map, (cx, cy), 8, (255, 255, 255), 1)

    return pitch_map

# ==============================================================================
# SECTION 3: TRACKING AND MULTI-PASS CLASSIFICATION ENGINE
# ==============================================================================
def process_match_lifecycle(video_path):
    from detect_gemini_v1 import get_homography_matrix
    H, scale_factor, canvas_width, canvas_height = get_homography_matrix(video_path)
    
    with open("camera_calibration.json", 'r') as f:
        calib_data = json.load(f)
    anchor_t1 = calib_data["team1_anchor_hsv"]
    anchor_t2 = calib_data["team2_anchor_hsv"]
    
    bgr_t1 = hsv_to_bgr(anchor_t1)
    bgr_t2 = hsv_to_bgr(anchor_t2)
    
    team1_bounds = (np.array([max(0, anchor_t1[0] - 14), 60, 50]), np.array([min(179, anchor_t1[0] + 14), 255, 255]))
    team2_bounds = (np.array([max(0, anchor_t2[0] - 14), 60, 50]), np.array([min(179, anchor_t2[0] + 14), 255, 255]))
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if TRACKING_WINDOW_FRAMES == 0:
        print("[+] 'TRACKING_WINDOW_FRAMES' is set to 0: Processing the entire video lifecycle.")
        start_frame = 0
        actual_processing_frames = total_frames
    else:
        min_start_frame = int(total_frames * 0.25)
        max_start_frame = int(total_frames * 0.75)
        if total_frames <= TRACKING_WINDOW_FRAMES:
            start_frame = 0
            actual_processing_frames = total_frames
        else:
            start_frame = random.randint(min_start_frame, max_start_frame)
            if start_frame + TRACKING_WINDOW_FRAMES > total_frames:
                start_frame = total_frames - TRACKING_WINDOW_FRAMES - 1
            actual_processing_frames = TRACKING_WINDOW_FRAMES
        print(f"[+] Processing Snippet Window: Frame {start_frame} to {start_frame + actual_processing_frames}")
        
    total_steps = int(actual_processing_frames // FRAME_SKIP_STEP)
    model = YOLO(YOLO_MODEL_WEIGHTS).to("mps")
    
    cushion_px = int(FIELD_CUSHION_METERS * scale_factor)
    padded_polygon = np.array([
        [-cushion_px, -cushion_px],
        [canvas_width + cushion_px, -cushion_px],
        [canvas_width + cushion_px, canvas_height + cushion_px],
        [-cushion_px, canvas_height + cushion_px]
    ], dtype=np.int32)

    output_fps = float(fps) / FRAME_SKIP_STEP
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer_camera = cv2.VideoWriter(OUTPUT_CAMERA_VIDEO, fourcc, output_fps, (frame_width, frame_height))
    writer_radar = cv2.VideoWriter(OUTPUT_RADAR_VIDEO, fourcc, output_fps, (canvas_width + 40, canvas_height + 40))

    player_count_timeline = []
    previous_tracked_players = {}
    next_track_id = 0

    # Start benchmarking wall-clock time
    start_time_perf = time.time()

    for step_idx in range(total_steps):
        target_frame = start_frame + (step_idx * FRAME_SKIP_STEP)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, imgsz=1920, conf=PLAYER_CONF_BASE, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        player_detections = detections[detections.class_id == 0]

        raw_detected_positions = []
        bboxes_map = []
        
        for bbox in player_detections.xyxy:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2.0, y2 
            
            point = np.array([[[cx, cy]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, H)[0][0]
            canvas_x, canvas_y = transformed[0], transformed[1]
            
            if cv2.pointPolygonTest(padded_polygon, (float(canvas_x), float(canvas_y)), False) >= 0:
                mx, my = canvas_x / scale_factor, canvas_y / scale_factor
                raw_detected_positions.append((mx, my))
                bboxes_map.append(bbox)

        current_frame_tracked = {}
        active_match_players = []
        display_frame = frame.copy()

        unclaimed_indices = set(range(len(raw_detected_positions)))

        for pid, p_state in previous_tracked_players.items():
            if not unclaimed_indices:
                break
                
            prev_coords = p_state["coords"]
            best_idx = min(unclaimed_indices, key=lambda idx: np.hypot(raw_detected_positions[idx][0] - prev_coords[0], 
                                                                       raw_detected_positions[idx][1] - prev_coords[1]))
            best_mx, best_my = raw_detected_positions[best_idx]
            dist = np.hypot(best_mx - prev_coords[0], best_my - prev_coords[1])

            if dist < MAX_MATCH_DISTANCE_M:
                smoothed_mx = (TRACK_SMOOTHING_ALPHA * best_mx) + ((1.0 - TRACK_SMOOTHING_ALPHA) * prev_coords[0])
                smoothed_my = (TRACK_SMOOTHING_ALPHA * best_my) + ((1.0 - TRACK_SMOOTHING_ALPHA) * prev_coords[1])
                
                current_raw_score = classify_torso_color(frame, bboxes_map[best_idx], team1_bounds, team2_bounds)
                smoothed_score = (COLOR_SMOOTHING_ALPHA * current_raw_score) + ((1.0 - COLOR_SMOOTHING_ALPHA) * p_state["team_score"])
                
                current_frame_tracked[pid] = {"coords": (smoothed_mx, smoothed_my), "team_score": smoothed_score}
                active_match_players.append({"mx": smoothed_mx, "my": smoothed_my, "team_score": smoothed_score})
                unclaimed_indices.remove(best_idx)
                
                box_color = get_dynamic_team_color(smoothed_score, bgr_t1, bgr_t2)
                x1, y1, x2, y2 = bboxes_map[best_idx]
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

        for idx in unclaimed_indices:
            mx, my = raw_detected_positions[idx]
            initial_score = classify_torso_color(frame, bboxes_map[idx], team1_bounds, team2_bounds)
            
            current_frame_tracked[next_track_id] = {"coords": (mx, my), "team_score": initial_score}
            active_match_players.append({"mx": mx, "my": my, "team_score": initial_score})
            next_track_id += 1
            
            box_color = get_dynamic_team_color(initial_score, bgr_t1, bgr_t2)
            x1, y1, x2, y2 = bboxes_map[idx]
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

        previous_tracked_players = current_frame_tracked
        player_count_timeline.append(len(active_match_players))
        
        writer_camera.write(display_frame)
        radar_frame = generate_birds_eye(active_match_players, canvas_width, canvas_height, bgr_t1, bgr_t2)
        writer_radar.write(radar_frame)
        
        if step_idx % 10 == 0 or step_idx == total_steps - 1:
            print_progress(step_idx + 1, total_steps, phase_name="Tracking & Color-Smoothing")

    cap.release()
    writer_camera.release()
    writer_radar.release()
    
    # Calculate performance execution metrics
    elapsed_time = time.time() - start_time_perf
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
    
    # Frames per minute formula: (Total input frames spanned / elapsed execution seconds) * 60
    frames_processed = total_steps * FRAME_SKIP_STEP
    frames_per_minute = (frames_processed / elapsed_time) * 60 if elapsed_time > 0 else 0
    
    print(f"\n\n[=] PERFORMANCE REPORT [=]")
    print(f"    -> Total Processing Duration : {elapsed_minutes}m {elapsed_seconds}s")
    print(f"    -> Original Video Frames Spanned : {frames_processed} frames")
    print(f"    -> Core Processing Speed : {frames_per_minute:.1f} Frames Per Minute (FPM)")
    print(f"[+] Render completed successfully!\n")

    # ==============================================================================
    # STEP 4: SECOND-PASS VARIANCE CHECK TO IDENTIFY MATCH TRANSITIONS
    # ==============================================================================
    print("[+] Analyzing density timeline metrics...")
    effective_timeline_fps = float(fps) / FRAME_SKIP_STEP
    window_steps = int(WINDOW_SECONDS * effective_timeline_fps)
    cooldown_steps = int(COOLDOWN_SECONDS * effective_timeline_fps)
    
    transition_events = []
    last_transition_step = -cooldown_steps
    
    for i in range(window_steps, len(player_count_timeline) - window_steps):
        past_window = player_count_timeline[i - window_steps : i]
        future_window = player_count_timeline[i : i + window_steps]
        
        past_mean = np.mean(past_window)
        future_mean = np.mean(future_window)
        
        mean_delta = abs(future_mean - past_mean)
        combined_std = np.std(past_window + future_window)
        
        if mean_delta > VARIANCE_THRESHOLD and combined_std > 1.0:
            if (i - last_transition_step) > cooldown_steps:
                absolute_frame_index = start_frame + (i * FRAME_SKIP_STEP)
                timestamp_str = frames_to_timestamp(absolute_frame_index, fps)
                
                transition_events.append({
                    "absolute_frame_index": absolute_frame_index,
                    "timestamp": timestamp_str,
                    "avg_players_before": round(float(past_mean), 1),
                    "avg_players_after": round(float(future_mean), 1)
                })
                last_transition_step = i

    output_payload = {
        "video_source": os.path.basename(video_path),
        "snippet_start_frame": start_frame,
        "snippet_end_frame": start_frame + actual_processing_frames,
        "total_transitions_found": len(transition_events),
        "transitions": transition_events
    }
    
    with open(OUTPUT_JSON_LOG, 'w') as f:
        json.dump(output_payload, f, indent=4)
        
    print(f"[+] Transition logs compiled into '{OUTPUT_JSON_LOG}'")


if __name__ == "__main__":
    my_video = "game.mp4" 
    process_match_lifecycle(my_video)