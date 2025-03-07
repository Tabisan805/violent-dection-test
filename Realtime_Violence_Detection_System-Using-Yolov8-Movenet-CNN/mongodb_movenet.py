import cv2
import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import pygame
import threading
import time
import os
import urllib.request
import sys
from datetime import datetime
from pymongo import MongoClient
import uuid

print("Starting enhanced violence detection system with video recording...")

# MongoDB configuration
MONGODB_CONNECTION_STRING = "mongodb+srv://frustberg:Abhishek123@cluster1.oqmff.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
DB_NAME = "violence_detection"
COLLECTION_NAME = "recorded_incidents"

# Video recording configuration
RECORDING_CONFIG = {
    "output_directory": "recorded_incidents",  # Directory to save recorded videos
    "pre_incident_buffer": 5,                  # Seconds of video to save before incident
    "post_incident_buffer": 5,                 # Seconds to continue recording after violence stops
    "video_codec": "XVID",                     # Video codec for saving
    "video_fps": 20,                           # Frames per second for saved video
    "max_recording_time": 60,                  # Maximum recording time per incident (seconds)
}

# MoveNet model path and URL
MOVENET_MODEL_PATH = "movenet_lightning.tflite"
MOVENET_MODEL_URL = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"

# Configuration
CONFIG = {
    "violence_threshold": 0.7,       # Balance between sensitivity and false positives
    "yolo_confidence": 0.5,           # YOLO detection confidence threshold
    "sequence_length": 20,            # Default frames per sequence for violence model
    "frame_size": (224, 224),         # Default frame dimensions for violence model
    "motion_threshold": 0.4,          # Suspicious motion threshold
    "jerk_threshold": 0.4,            # Sudden movement threshold
    "alert_cooldown": 3,              # Seconds between alerts
    "debug_mode": True                # Enable detailed logging
}

# Create output directory for recorded videos if it doesn't exist
if not os.path.exists(RECORDING_CONFIG["output_directory"]):
    os.makedirs(RECORDING_CONFIG["output_directory"])
    print(f"Created directory for recorded incidents: {RECORDING_CONFIG['output_directory']}")

# Initialize MongoDB connection
try:
    mongo_client = MongoClient(MONGODB_CONNECTION_STRING)
    db = mongo_client[DB_NAME]
    incidents_collection = db[COLLECTION_NAME]
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    mongo_client = None
    db = None
    incidents_collection = None

# Check model files
model_missing = False

print("Starting enhanced violence detection system...")

# FIX: Download MoveNet model if not available
def download_movenet_model():
    try:
        if not os.path.exists(MOVENET_MODEL_PATH):
            print(f"MoveNet model not found. Downloading from {MOVENET_MODEL_URL}...")
            # Create a request with a fake user agent to avoid potential blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            req = urllib.request.Request(MOVENET_MODEL_URL, headers=headers)
            with urllib.request.urlopen(req) as response, open(MOVENET_MODEL_PATH, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print("MoveNet model downloaded successfully")
            return True
        else:
            print("MoveNet model already exists")
            return True
    except Exception as e:
        print(f"Error downloading MoveNet model: {e}")
        return False

# Try to download MoveNet model
movenet_available = download_movenet_model()

# Initialize pygame mixer for alert sound
pygame.mixer.init()
alert_sound = "alarm_sound.mp3"
if not os.path.exists(alert_sound):
    print("WARNING: Alarm sound file 'alarm_sound.mp3' not found.")

def play_alert():
    try:
        pygame.mixer.music.load(alert_sound)
        pygame.mixer.music.play(-1)  # Loop indefinitely
    except Exception as e:
        print(f"Error playing alert sound: {e}")
        print("\a")  # ASCII bell as fallback

def stop_alert():
    pygame.mixer.music.stop()

# Alert status tracking
alert_active = False

def update_alert_status(detected):
    global alert_active
    if detected and not alert_active:
        alert_active = True
        threading.Thread(target=play_alert, daemon=True).start()
    elif not detected and alert_active:
        alert_active = False
        stop_alert()

# Check for violence detection model
if not os.path.exists('violence_detection_model.h5'):
    print("ERROR: Violence detection model 'violence_detection_model.h5' not found.")
    model_missing = True

# Check for YOLOv8 model (will download automatically if missing)
try:
    yolo_model = YOLO("yolov8n.pt")
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model_missing = True

# Check for alarm sound file
if not os.path.exists('alarm_sound.mp3'):
    print("WARNING: Alarm sound file 'alarm_sound.mp3' not found. Creating a default sound file.")
    try:
        pygame.mixer.init()
        pygame.mixer.Sound(np.sin(2 * np.pi * np.arange(44100) * 440 / 44100).astype(np.float32)).save('alarm_sound.mp3')
        print("Created default alarm sound")
    except Exception as e:
        print(f"Could not create default alarm sound: {e}")

# FIX: Load violence detection model with better error handling
violence_model = None
try:
    # Try using a custom loader approach
    model_path = 'violence_detection_model.h5'
    if os.path.exists(model_path):
        try:
            # First attempt: Using direct load with compile=False
            violence_model = load_model(model_path, compile=False)
        except Exception as e1:
            print(f"First loading attempt failed: {e1}")
            try:
                # Second attempt: Using custom objects with None for batch_shape
                custom_objects = {'batch_shape': None}
                violence_model = load_model(model_path, compile=False, custom_objects=custom_objects)
            except Exception as e2:
                print(f"Second loading attempt failed: {e2}")
                try:
                    # Third attempt: Load with tf.keras directly
                    violence_model = tf.keras.models.load_model(model_path, compile=False)
                except Exception as e3:
                    print(f"Third loading attempt failed: {e3}")
                    violence_model = None
    
        if violence_model is not None:
            print("Violence detection model loaded successfully")
            
            # Try to get model info
            try:
                input_shape = violence_model.input_shape
                print(f"Model expects input shape: {input_shape}")
                
                # Extract dimensions from the model
                if input_shape and len(input_shape) >= 4:
                    if len(input_shape) == 5:  # 3D CNN
                        _, SEQ_LENGTH, HEIGHT, WIDTH, CHANNELS = input_shape
                        CONFIG["frame_size"] = (WIDTH, HEIGHT)
                        CONFIG["sequence_length"] = SEQ_LENGTH
                    else:  # 2D CNN
                        _, HEIGHT, WIDTH, CHANNELS = input_shape
                        CONFIG["frame_size"] = (WIDTH, HEIGHT)
                    
                    print(f"Updated configuration based on model: {CONFIG['frame_size']}, sequence_length: {CONFIG['sequence_length']}")
            except Exception as e:
                print(f"Error determining model input shape: {e}")
    else:
        print(f"Violence detection model file not found at {model_path}")
except Exception as e:
    print(f"Error loading violence detection model: {e}")
    print("Continuing with motion-based detection only")

# Initialize camera
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 20)  # Aim for 20 FPS for more stable processing
    print("Camera initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize camera: {e}")
    sys.exit(1)

# Get camera properties for recording
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = RECORDING_CONFIG["video_fps"]  # Use default if camera doesn't report FPS

# Initialize frame storage
frames = deque(maxlen=CONFIG["sequence_length"])
PREDICTION_HISTORY = deque(maxlen=15)  # Store for smoother predictions
POSE_HISTORY = deque(maxlen=10)  # Store pose keypoints for motion analysis

# Initialize frame buffer for pre-incident recording
pre_incident_frames = deque(maxlen=int(RECORDING_CONFIG["pre_incident_buffer"] * fps))

# Video recording state variables
is_recording = False
video_writer = None
recording_start_time = 0
current_recording_path = None
current_recording_id = None
violence_end_time = 0

# IMPROVED: Load model into memory and validate for inference
def validate_model(model, expected_shape):
    """Test if model can run inference with expected shape."""
    if model is None:
        return False
    
    try:
        # Create a dummy input with the expected shape
        dummy_input = np.zeros(expected_shape)
        # Try to predict with the dummy input
        model.predict(dummy_input, verbose=0)
        return True
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False

# IMPROVED: Create model wrapper to handle different input formats
class ModelWrapper:
    def __init__(self, model, sequence_length=20, frame_size=(224, 224)):
        self.model = model
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.is_compatible = False
        
        if model is not None:
            # Validate model compatibility
            expected_shape = (1, sequence_length, frame_size[1], frame_size[0], 3)
            self.is_compatible = validate_model(model, expected_shape)
            
            if not self.is_compatible:
                print("Model is incompatible with expected shape.")
                try:
                    # Try to get model info
                    input_shape = model.input_shape
                    if input_shape and len(input_shape) == 5:
                        _, seq, height, width, _ = input_shape
                        self.sequence_length = seq
                        self.frame_size = (width, height)
                        print(f"Updated model parameters: {self.frame_size}, sequence_length: {self.sequence_length}")
                        
                        # Try validation again with proper shape
                        expected_shape = (1, self.sequence_length, self.frame_size[1], self.frame_size[0], 3)
                        self.is_compatible = validate_model(model, expected_shape)
                        if self.is_compatible:
                            print("Model is now compatible with adjusted parameters.")
                except Exception as e:
                    print(f"Error examining model: {e}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        # Resize to expected dimensions
        frame = cv2.resize(frame, self.frame_size)
        # Convert to RGB (model was trained on RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to [0, 1]
        frame = frame.astype('float32') / 255.0
        return frame
    
    def predict(self, frames_list):
        """Run prediction using proper sequence format."""
        if not self.is_compatible or self.model is None:
            return 0.0
        
        if len(frames_list) != self.sequence_length:
            return 0.0
            
        try:
            # Convert frames to numpy array
            sequence = np.array(frames_list)
            # Add batch dimension
            sequence = np.expand_dims(sequence, axis=0)
            # Run prediction
            pred = self.model.predict(sequence, verbose=0)[0][0]
            return float(pred)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

# IMPROVED: Initialize model wrapper with validation
model_wrapper = ModelWrapper(
    violence_model,
    sequence_length=CONFIG["sequence_length"],
    frame_size=CONFIG["frame_size"]
)

# Update config based on model compatibility
if model_wrapper.is_compatible:
    CONFIG["sequence_length"] = model_wrapper.sequence_length
    CONFIG["frame_size"] = model_wrapper.frame_size
    print(f"Using validated model parameters: {CONFIG['frame_size']}, sequence_length: {CONFIG['sequence_length']}")
    # Reinitialize frame storage with correct sequence length
    frames = deque(maxlen=CONFIG["sequence_length"])

# Alert function
def play_alert():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alarm_sound.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Error playing alert sound: {e}")
        print("\a")  # ASCII bell as fallback

# FIX: Initialize MoveNet using TF Lite with proper error handling
interpreter = None
if movenet_available:
    try:
        # Check if file exists and is not empty
        if os.path.exists(MOVENET_MODEL_PATH) and os.path.getsize(MOVENET_MODEL_PATH) > 0:
            interpreter = tf.lite.Interpreter(model_path=MOVENET_MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("MoveNet model loaded successfully")
        else:
            print("MoveNet model file exists but may be corrupted. Will continue without pose estimation.")
            movenet_available = False
    except Exception as e:
        print(f"Error loading MoveNet model: {e}")
        print("Will continue without pose estimation")
        movenet_available = False

# IMPROVED: Run MoveNet pose estimation using TFLite with better error handling
def detect_pose(image):
    if interpreter is None or not movenet_available:
        return np.zeros((17, 3))  # Return empty pose if model not loaded
    
    try:
        # Resize and normalize the image
        input_image = cv2.resize(image, (192, 192))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        
        # Convert to uint8 instead of float32
        input_image = (input_image * 255).astype(np.uint8)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output
        keypoints = interpreter.get_tensor(output_details[0]['index'])
        keypoints = keypoints.reshape((17, 3))  # 17 keypoints, each with [y, x, confidence]
        
        # Convert normalized coordinates to pixel values
        img_height, img_width = image.shape[0], image.shape[1]
        for i in range(keypoints.shape[0]):
            keypoints[i, 1] = keypoints[i, 1] * img_width  # x coordinate
            keypoints[i, 0] = keypoints[i, 0] * img_height  # y coordinate
        
        # Rearrange to [x, y, confidence] format
        result = np.zeros((17, 3))
        result[:, 0] = keypoints[:, 1]  # x
        result[:, 1] = keypoints[:, 0]  # y
        result[:, 2] = keypoints[:, 2]  # confidence
        
        return result
    except Exception as e:
        print(f"Error in pose detection: {e}")
        return np.zeros((17, 3))  # Return empty pose on error

# FIX: Fallback motion analysis for when pose estimation is unavailable
def analyze_frame_motion(current_frame, prev_frame):
    """Analyze motion between frames without pose estimation"""
    if prev_frame is None or current_frame is None:
        return 0.0
    
    try:
        # Convert to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude and angle of flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get motion statistics
        mean_motion = np.mean(magnitude)
        max_motion = np.max(magnitude)
        
        # Normalize motion score (0-1 range)
        motion_score = min(mean_motion / 10.0, 1.0) * 0.5 + min(max_motion / 50.0, 1.0) * 0.5
        
        return motion_score
    except Exception as e:
        print(f"Error in frame motion analysis: {e}")
        return 0.0

# IMPROVED: Analyze motion patterns for violence detection with better metrics
def analyze_motion(current_pose):
    if len(POSE_HISTORY) < 2:
        POSE_HISTORY.append(current_pose)
        return 0.0
    
    # Calculate movement metrics
    prev_pose = POSE_HISTORY[-1]
    
    # 1. Calculate velocity (movement between frames)
    velocities = []
    for i in range(len(current_pose)):
        if current_pose[i, 2] > 0.2 and prev_pose[i, 2] > 0.2:  # Only use confident keypoints
            dx = current_pose[i, 0] - prev_pose[i, 0]
            dy = current_pose[i, 1] - prev_pose[i, 1]
            velocity = np.sqrt(dx**2 + dy**2)  # Fixed the calculation (was dx*2 + dy*2)
            velocities.append(velocity)
    
    # 2. Calculate jerk (sudden changes in velocity)
    jerks = []
    if len(POSE_HISTORY) >= 3:
        prev_prev_pose = POSE_HISTORY[-2]
        for i in range(len(current_pose)):
            if (current_pose[i, 2] > 0.2 and prev_pose[i, 2] > 0.2 and 
                prev_prev_pose[i, 2] > 0.2):
                # Previous velocity
                prev_dx = prev_pose[i, 0] - prev_prev_pose[i, 0]
                prev_dy = prev_pose[i, 1] - prev_prev_pose[i, 1]
                prev_velocity = np.sqrt(prev_dx**2 + prev_dy**2)  # Fixed calculation
                
                # Current velocity
                curr_dx = current_pose[i, 0] - prev_pose[i, 0]
                curr_dy = current_pose[i, 1] - prev_pose[i, 1]
                curr_velocity = np.sqrt(curr_dx**2 + curr_dy**2)  # Fixed calculation
                
                # Jerk (change in velocity)
                jerk = abs(curr_velocity - prev_velocity)
                jerks.append(jerk)
    
    # 3. IMPROVED: Detect fighting or aggressive stances
    aggressive_pose_score = 0.0
    
    # Check for raised arms (possible fighting stance)
    if current_pose[5, 2] > 0.2 and current_pose[6, 2] > 0.2:  # Shoulders
        shoulder_y = (current_pose[5, 1] + current_pose[6, 1]) / 2
        
        # Check for raised wrists above shoulders
        if current_pose[9, 2] > 0.2 and current_pose[9, 1] < shoulder_y:  # Left wrist above shoulder
            aggressive_pose_score += 0.25
        
        if current_pose[10, 2] > 0.2 and current_pose[10, 1] < shoulder_y:  # Right wrist above shoulder
            aggressive_pose_score += 0.25
    
    # Check for arms extended outward (possible punching/pushing)
    if (current_pose[5, 2] > 0.2 and current_pose[9, 2] > 0.2 and  # Left shoulder and wrist
        current_pose[6, 2] > 0.2 and current_pose[10, 2] > 0.2):   # Right shoulder and wrist
        
        left_arm_extension = np.sqrt((current_pose[9, 0] - current_pose[5, 0])**2 + 
                                     (current_pose[9, 1] - current_pose[5, 1])**2)
        right_arm_extension = np.sqrt((current_pose[10, 0] - current_pose[6, 0])**2 + 
                                      (current_pose[10, 1] - current_pose[6, 1])**2)
        
        # Normalize by shoulder width
        shoulder_width = np.sqrt((current_pose[5, 0] - current_pose[6, 0])**2 + 
                                (current_pose[5, 1] - current_pose[6, 1])**2)
        
        if shoulder_width > 0:
            left_extension_ratio = left_arm_extension / shoulder_width
            right_extension_ratio = right_arm_extension / shoulder_width
            
            # Extended arms (possible hitting motion)
            if left_extension_ratio > 1.8:
                aggressive_pose_score += 0.25
            if right_extension_ratio > 1.8:
                aggressive_pose_score += 0.25
    
    # Store current pose for next comparison
    POSE_HISTORY.append(current_pose)
    
    # IMPROVED: Combine metrics for motion violence score with better weighting
    motion_score = 0.0
    
    # Process velocity information
    if velocities:
        max_velocity = max(velocities)
        # Normalize velocity with improved scaling
        norm_velocity = min(max_velocity / 30, 1.0)
        motion_score += norm_velocity * 0.4  # 40% weight
        
        # Print high velocity for debugging
        if max_velocity > 20 and CONFIG["debug_mode"]:
            print(f"High velocity detected: {max_velocity:.2f}, normalized: {norm_velocity:.2f}")
    
    # Process jerk information (sudden movements)
    if jerks:
        max_jerk = max(jerks)
        # Normalize jerk with improved scaling
        norm_jerk = min(max_jerk / 15, 1.0)
        motion_score += norm_jerk * 0.4  # 40% weight
        
        # Print high jerk for debugging
        if max_jerk > 10 and CONFIG["debug_mode"]:
            print(f"High jerk detected: {max_jerk:.2f}, normalized: {norm_jerk:.2f}")
    
    # Add aggressive pose contribution
    motion_score += aggressive_pose_score * 0.2  # 20% weight
    
    # Debug: high motion score analysis
    if motion_score > 0.3 and CONFIG["debug_mode"]:
        print(f"Motion score: {motion_score:.2f} (velocities: {len(velocities)}, jerks: {len(jerks)}, aggressive pose: {aggressive_pose_score:.2f}")
    
    return motion_score

# IMPROVED: Weighted smoothing for prediction stability with more advanced algorithm
def get_smoothed_prediction(prediction):
    PREDICTION_HISTORY.append(prediction)
    
    # Apply exponential weighting (recent predictions matter more)
    weights = np.exp(np.linspace(0, 1, len(PREDICTION_HISTORY)))
    weighted_avg = np.average(PREDICTION_HISTORY, weights=weights)
    
    # Apply minor hysteresis to prevent oscillation around threshold
    if len(PREDICTION_HISTORY) > 5:
        recent_avg = np.mean(list(PREDICTION_HISTORY)[-5:])
        prev_avg = np.mean(list(PREDICTION_HISTORY)[:-5])
        
        # If trending upward, boost slightly; if trending downward, reduce slightly
        if recent_avg > prev_avg:
            weighted_avg = weighted_avg * 1.05  # Boost upward trend
        else:
            weighted_avg = weighted_avg * 0.95  # Suppress downward trend
    
    return min(weighted_avg, 1.0)  # Cap at 1.0

# NEW: Function to start recording when violence is detected
def start_recording(frame):
    global is_recording, video_writer, recording_start_time, current_recording_path, current_recording_id
    
    # Don't start new recording if already recording
    if is_recording:
        return
    
    # Generate unique ID for this recording
    current_recording_id = str(uuid.uuid4())
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"violence_{timestamp}_{current_recording_id[:8]}.avi"
    current_recording_path = os.path.join(RECORDING_CONFIG["output_directory"], filename)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        current_recording_path,
        fourcc,
        RECORDING_CONFIG["video_fps"],
        (frame_width, frame_height)
    )
    
    # Write pre-incident buffer frames first
    for buffered_frame in pre_incident_frames:
        video_writer.write(buffered_frame)
    
    # Record metadata
    recording_start_time = time.time()
    is_recording = True
    
    print(f"Started recording: {current_recording_path}")
    
    # Insert record into MongoDB
    if incidents_collection is not None:
        try:
            incident_record = {
                "incident_id": current_recording_id,
                "filename": filename,
                "filepath": current_recording_path,
                "timestamp_start": datetime.now(),
                "detected_score": smoothed_prediction,
                "status": "recording"
            }
            incidents_collection.insert_one(incident_record)
            print(f"Created MongoDB record for incident: {current_recording_id}")
        except Exception as e:
            print(f"Error creating MongoDB record: {e}")

# NEW: Function to stop recording
def stop_recording():
    global is_recording, video_writer, current_recording_path, current_recording_id
    
    if not is_recording:
        return
    
    # Close video writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    
    recording_duration = time.time() - recording_start_time
    print(f"Stopped recording after {recording_duration:.2f} seconds: {current_recording_path}")
    
    # Update MongoDB record
    if incidents_collection is not None and current_recording_id is not None:
        try:
            incidents_collection.update_one(
                {"incident_id": current_recording_id},
                {
                    "$set": {
                        "timestamp_end": datetime.now(),
                        "duration_seconds": recording_duration,
                        "status": "completed",
                        "file_size_kb": os.path.getsize(current_recording_path) / 1024
                    }
                }
            )
            print(f"Updated MongoDB record for incident: {current_recording_id}")
        except Exception as e:
            print(f"Error updating MongoDB record: {e}")
    
    is_recording = False
    current_recording_path = None
    current_recording_id = None

# Main processing loop
smoothed_prediction = 0.0
motion_score = 0.0
frame_count = 0
alert_triggered = False
last_alert_time = 0
violence_detected = False
violence_stopped_time = 0
prev_frame = None  # For frame-based motion detection

try:
    print("Press 'q' to quit, 't' to test violence detection, 'r' to manually start/stop recording")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue

        frame_count += 1
        current_time = time.time()

        # Add the frame to pre-incident buffer
        pre_incident_frames.append(frame.copy())
        
        # Frame skipping for performance (process every 2nd frame)
        if frame_count % 2 != 0:
            if is_recording and video_writer is not None:
                video_writer.write(frame)
            continue
        
        # Make a copy for display
        display_frame = frame.copy()
        
        # Run YOLOv8 object detection
        try:
            results = yolo_model(frame)
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            continue
        
        # Detect people and focus on them
        detected_humans = []
        human_boxes = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])

                # Assuming YOLO class '0' is 'person'
                if cls == 0 and conf > CONFIG["yolo_confidence"]:
                    # Ensure box coordinates are valid
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    # Only add if the box has valid dimensions
                    if x2 > x1 and y2 > y1:
                        human_frame = frame[y1:y2, x1:x2]
                        detected_humans.append(human_frame)
                        human_boxes.append((x1, y1, x2, y2))
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Person: {conf:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Initialize violence predictions for this frame
        this_frame_prediction = 0.0
        
        # Process detected humans for violence detection
        if detected_humans:
            # Use the largest detected human for pose estimation
            largest_human_idx = np.argmax([h.shape[0] * h.shape[1] for h in detected_humans])
            human_frame = detected_humans[largest_human_idx]
            human_box = human_boxes[largest_human_idx]
            
            # Run pose detection
            if movenet_available and interpreter is not None:
                pose_keypoints = detect_pose(human_frame)
                
                # Analyze motion using pose data
                motion_score = analyze_motion(pose_keypoints)
                
                # Draw pose keypoints on display frame
                for i in range(pose_keypoints.shape[0]):
                    if pose_keypoints[i, 2] > 0.2:  # Only show confident keypoints
                        x, y = int(pose_keypoints[i, 0] + human_box[0]), int(pose_keypoints[i, 1] + human_box[1])
                        cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            else:
                # Fallback to frame-by-frame motion analysis
                motion_score = analyze_frame_motion(human_frame, prev_frame)
                prev_frame = human_frame.copy()
            
            # Update motion score display
            motion_text = f"Motion: {motion_score:.2f}"
            cv2.putText(display_frame, motion_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Process with violence detection model if available
            if model_wrapper.is_compatible:
                # Preprocess frame
                processed_frame = model_wrapper.preprocess_frame(human_frame)
                frames.append(processed_frame)
                
                # Run model prediction if we have enough frames
                if len(frames) == CONFIG["sequence_length"]:
                    # Get model prediction
                    try:
                        this_frame_prediction = model_wrapper.predict(list(frames))
                    except Exception as e:
                        print(f"Error in model prediction: {e}")
                        this_frame_prediction = 0.0
            else:
                # Use motion-based detection only
                this_frame_prediction = motion_score
        
        # Get smoothed prediction score
        smoothed_prediction = get_smoothed_prediction(this_frame_prediction)
        
        # Determine if violence is detected
        violence_detected = smoothed_prediction > CONFIG["violence_threshold"]
        
        # Update recording state based on violence detection
        if violence_detected:
            violence_stopped_time = 0  # Reset cooldown timer
            if not is_recording:
                start_recording(frame)
        elif is_recording:
            # Start cooldown for stopping recording
            if violence_stopped_time == 0:
                violence_stopped_time = current_time
            
            # Stop recording after post-incident buffer time
            if current_time - violence_stopped_time > RECORDING_CONFIG["post_incident_buffer"]:
                stop_recording()
            
            # Also stop if maximum recording time is reached
            if current_time - recording_start_time > RECORDING_CONFIG["max_recording_time"]:
                stop_recording()
        
        # Handle alert sound based on detection
        if violence_detected:
            if not alert_triggered or (current_time - last_alert_time > CONFIG["alert_cooldown"]):
                update_alert_status(True)
                alert_triggered = True
                last_alert_time = current_time
                print(f"⚠️ VIOLENCE DETECTED! Score: {smoothed_prediction:.2f}")
        else:
            if alert_triggered:
                update_alert_status(False)
                alert_triggered = False
        
        # Display status information
        status_color = (0, 0, 255) if violence_detected else (0, 255, 0)
        status_text = "VIOLENCE DETECTED" if violence_detected else "Monitoring"
        cv2.putText(display_frame, status_text, (10, frame_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display prediction score
        score_text = f"Score: {smoothed_prediction:.2f}"
        cv2.putText(display_frame, score_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display recording status if recording
        if is_recording:
            rec_time = time.time() - recording_start_time
            cv2.putText(display_frame, f"REC {rec_time:.1f}s", (frame_width - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add red recording circle
            cv2.circle(display_frame, (frame_width - 170, 25), 10, (0, 0, 255), -1)
        
        # Draw threshold line
        threshold_y = int((1 - CONFIG["violence_threshold"]) * 100) + 50
        cv2.line(display_frame, (150, threshold_y), (300, threshold_y), (0, 255, 255), 2)
        
        # Draw prediction bar
        bar_height = int((1 - smoothed_prediction) * 100) + 50
        bar_color = (0, 0, 255) if violence_detected else (0, 255, 0)
        cv2.rectangle(display_frame, (150, bar_height), (300, 150), bar_color, -1)
        cv2.rectangle(display_frame, (150, 50), (300, 150), (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Violence Detection System", display_frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            print("Exiting...")
            break
            
        # 't' to test detection
        elif key == ord('t'):
            print("Test: Simulating violence detection")
            this_frame_prediction = 0.9
            
        # 'r' to manually toggle recording
        elif key == ord('r'):
            if not is_recording:
                print("Manual recording started")
                start_recording(frame)
            else:
                print("Manual recording stopped")
                stop_recording()

except KeyboardInterrupt:
    print("Keyboard interrupt. Exiting...")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # Cleanup
    if is_recording:
        stop_recording()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Close MongoDB connection
    if mongo_client is not None:
        mongo_client.close()
        print("MongoDB connection closed")
    
    print("System shutdown complete")

# Additional utility functions

def get_system_status():
    """Get the current status of the system for logging or API purposes"""
    status = {
        "system_running": True,
        "violence_detected": violence_detected,
        "current_score": smoothed_prediction,
        "recording_active": is_recording,
        "models_loaded": {
            "yolo": yolo_model is not None,
            "violence_detection": model_wrapper.is_compatible,
            "pose_estimation": movenet_available and interpreter is not None
        },
        "config": CONFIG,
        "recording_config": RECORDING_CONFIG
    }
    
    # Add recording info if active
    if is_recording:
        status["recording"] = {
            "id": current_recording_id,
            "path": current_recording_path,
            "duration": time.time() - recording_start_time
        }
    
    return status

# If you want to run this file as a module in another application
if __name__ == "__main__":
    print("Violence detection system completed execution")