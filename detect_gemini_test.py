import random
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Assuming these are accessible from your calibration script setup
from detect_gemini_v1 import get_homography_matrix, get_metric_coordinates

def sample_torso_color(frame, bbox):
    """Crops the middle-top area of a player bounding box to extract their vest color."""
    x1, y1, x2, y2 = map(int, bbox)
    w = x2 - x1
    h = y2 - y1
    
    # Crop the torso area (vertical 20% to 50% of the box)
    torso = frame[int(y1 + h * 0.2):int(y1 + h * 0.5), int(x1 + w * 0.1):int(x2 - w * 0.1)]
    
    if torso.size == 0:
        return np.array([128, 128, 128]) # Fallback neutral grey if crop fails
        
    # Calculate average BGR color of the torso
    return torso.mean(axis=0).mean(axis=0)

import detect_gemini_v1  # Make sure you are importing the module like this

def inspect_random_match_frame(video_path):
    # 1. Load Calibration Parameters
    H, scale_factor, canvas_width, canvas_height = get_homography_matrix(video_path)
    
    # FORCE INJECTION: Dynamically assign SCALE_FACTOR into the calibration file's namespace
    detect_gemini_v1.SCALE_FACTOR = scale_factor
    
    # 2. Pick a Random Frame between 25% and 75% of the Video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(total_frames * 0.25)
    end_frame = int(total_frames * 0.75)
    random_target_frame = random.randint(start_frame, end_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[-] Error: Could not read the selected random frame.")
        return

    print(f"[+] Loaded Random Frame: {random_target_frame} / {total_frames}")
    
    
    # 3. Run YOLO Inference
    model = YOLO("yolov8m.pt")
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter for Person (Class 0) and Ball (Class 32)
    valid_classes = (detections.class_id == 0) | (detections.class_id == 32)
    detections = detections[valid_classes]
    
# 4. Map Detections into Meter Coordinates & Separate by Halves (Strictly Filtered)
    players_list = []
    ball_bbox = None
    
    top_colors = []
    bottom_colors = []
    
    # Define your pitch boundary constants explicitly here if not global
    FIELD_WIDTH_M = 25.0
    FIELD_HEIGHT_M = 45.0
    MIDFIELD_LINE_M = FIELD_HEIGHT_M / 2.0  # 22.5m
    
    for idx, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = y2 if class_id == 0 else (y1 + y2) / 2.0  # Feet for players, center for ball
        
        # Transform pixels to real-world meters
        mx, my = get_metric_coordinates(cx, cy, H)
        
        # --- THE SPATIAL GUARDRAIL CRITICAL FIX ---
        # Look at the real-world dimensions: width must be 0-25m, height must be 0-45m
        if not (0.0 <= mx <= FIELD_WIDTH_M and 0.0 <= my <= FIELD_HEIGHT_M):
            continue  # Completely ignore this person/noise; they are on another pitch!
            
        if class_id == 32:
            ball_bbox = bbox
            continue
            
        # Extract individual torso color signature only for valid in-bounds players
        torso_color = sample_torso_color(frame, bbox)
        
        player_data = {
            "index": idx, 
            "bbox": bbox, 
            "mx": mx, 
            "my": my,
            "raw_color": torso_color
        }
        players_list.append(player_data)
        
        # Collect colors to determine the overall average team colors per half
        if my < MIDFIELD_LINE_M:
            top_colors.append(torso_color)
        else:
            bottom_colors.append(torso_color)


    # 5. Compute Dynamic Team Colors (Fall back to default if half is empty)
    team_top_color = tuple(map(int, np.mean(top_colors, axis=0))) if top_colors else (255, 0, 0)
    team_bottom_color = tuple(map(int, np.mean(bottom_colors, axis=0))) if bottom_colors else (0, 255, 255)

    # 6. Identify the Two Goalies Based on Proximity to Goal Targets
    goal_a_target = np.array([12.5, 0.0])   # Top Goal Center
    goal_b_target = np.array([12.5, 45.0])  # Bottom Goal Center
    
    goalie_a_idx = None
    goalie_b_idx = None
    min_dist_a = float('inf')
    min_dist_b = float('inf')
    
    for p in players_list:
        pos = np.array([p["mx"], p["my"]])
        dist_a = np.linalg.norm(pos - goal_a_target)
        dist_b = np.linalg.norm(pos - goal_b_target)
        
        if dist_a < min_dist_a:
            min_dist_a = dist_a
            goalie_a_idx = p["index"]
            
        if dist_b < min_dist_b:
            min_dist_b = dist_b
            goalie_b_idx = p["index"]

# [THIS REPLACES SECTIONS 7 & 8 IN YOUR CURRENT VALIDATION SCRIPT]
    
    # 7. Draw Predictions and Labels onto the Camera Frame
    display_frame = frame.copy()
    
    COLOR_BLUE   = (255, 50, 50)
    COLOR_YELLOW = (0, 240, 240)
    COLOR_PINK   = (255, 0, 255)
    COLOR_GREEN  = (0, 255, 0)
    COLOR_ORANGE = (0, 140, 255)

    if ball_bbox is not None:
        x1, y1, x2, y2 = map(int, ball_bbox)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLOR_ORANGE, 2)
        cv2.putText(display_frame, "Ball", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ORANGE, 2)

    for p in players_list:
        idx = p["index"]
        x1, y1, x2, y2 = map(int, p["bbox"])
        mx, my = p["mx"], p["my"]
        
        if idx == goalie_a_idx:
            box_color = COLOR_PINK
            label = f"Goalie Top ({mx:.1f}m, {my:.1f}m)"
        elif idx == goalie_b_idx:
            box_color = COLOR_GREEN
            label = f"Goalie Bottom ({mx:.1f}m, {my:.1f}m)"
        elif my < 22.5:
            box_color = COLOR_BLUE
            label = f"Team Blue ({mx:.1f}m, {my:.1f}m)"
        else:
            box_color = COLOR_YELLOW
            label = f"Team Yellow ({mx:.1f}m, {my:.1f}m)"
            
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 2)
        
    # 8. NEW: Generate the Bird's-Eye View Radar Map
    birds_eye_map = generate_tactical_birds_eye(
        players_list, ball_bbox, H, canvas_width, canvas_height, goalie_a_idx, goalie_b_idx
    )
        
    # 9. Render both windows to screen and save the results
    cv2.imshow("1. Camera View (Bounded & Filtered)", display_frame)
    cv2.imshow("2. Bird's-Eye 2D Tactical View", birds_eye_map)
    
    # Save copies to your project directory for debugging
    cv2.imwrite("output_camera_view.jpg", display_frame)
    cv2.imwrite("output_birds_eye_view.jpg", birds_eye_map)
    
    print("[+] Evaluation complete! Saved files:")
    print("    - output_camera_view.jpg")
    print("    - output_birds_eye_view.jpg")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def generate_tactical_birds_eye(players_list, ball_bbox, H, canvas_width, canvas_height, goalie_a_idx, goalie_b_idx):
    """
    Generates a clean, top-down 25x45m tactical pitch map 
    and plots the calculated positions of all in-bounds entities.
    """
    # 1. Initialize our pitch canvas (Green grass)
    pitch_map = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    pitch_map[:] = (40, 130, 40) 
    
    # 2. Draw standard vertical field markings (Scaled to canvas)
    cv2.rectangle(pitch_map, (0, 0), (canvas_width - 1, canvas_height - 1), (255, 255, 255), 2)
    
    mid_y = canvas_height // 2
    cv2.line(pitch_map, (0, mid_y), (canvas_width, mid_y), (255, 255, 255), 2) # Midfield
    
    scale_factor = canvas_width // 25 # Width is 25m
    cv2.circle(pitch_map, (canvas_width // 2, mid_y), int(6 * scale_factor), (255, 255, 255), 2) # Center Circle

    # Hyper-visible color profiles (Matching the video overlays)
    COLOR_BLUE   = (255, 50, 50)
    COLOR_YELLOW = (0, 240, 240)
    COLOR_PINK   = (255, 0, 255)
    COLOR_GREEN  = (0, 255, 0)
    
    # 3. Plot the Players
    for p in players_list:
        # Convert their meter positions back to canvas pixel coordinates
        cx_canvas = int(p["mx"] * scale_factor)
        cy_canvas = int(p["my"] * scale_factor)
        
        # Determine dot color
        if p["index"] == goalie_a_idx:
            dot_color = COLOR_PINK
        elif p["index"] == goalie_b_idx:
            dot_color = COLOR_GREEN
        elif p["my"] < 22.5:
            dot_color = COLOR_BLUE
        else:
            dot_color = COLOR_YELLOW
            
        # Draw player dot and their tracking ID number
        cv2.circle(pitch_map, (cx_canvas, cy_canvas), 8, dot_color, -1)
        cv2.circle(pitch_map, (cx_canvas, cy_canvas), 8, (255, 255, 255), 1) # White border
        cv2.putText(pitch_map, str(p["index"]), (cx_canvas - 5, cy_canvas + 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    return pitch_map
        
if __name__ == "__main__":
    # Put your actual video file name here!
    my_video = "game.mp4" 
    
    # Run the validation function we just built
    inspect_random_match_frame(my_video)