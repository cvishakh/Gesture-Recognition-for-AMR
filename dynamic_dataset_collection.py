# Import libraries
import cv2
import numpy as np
import os
import time
import pyzed.sl as sl
import mediapipe as mp

# Define gesture classes (Pointing = 4, Waving = 5)
GESTURE_CLASSES = {4: "pointing", 5: "waving"}

# Path for dynamic dataset
DYNAMIC_DATASET_PATH = "dataset\dynamic"

# Initialize ZED 2i Camera
zed = sl.Camera()
init_params = sl.InitParameters(
    depth_mode=sl.DEPTH_MODE.PERFORMANCE,
    coordinate_units=sl.UNIT.METER,
    camera_resolution=sl.RESOLUTION.HD720
)

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("ZED 2i Camera failed to initialize!")
    exit(1)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# User input for gesture collection
gesture_label = input("Enter the gesture label (4: Pointing, 5: Waving): ")

# Ensure gesture label is valid
if int(gesture_label) not in GESTURE_CLASSES:
    print("Invalid gesture label! Please enter 4 or 5.")
    exit(1)

gesture_name = GESTURE_CLASSES[int(gesture_label)]
gesture_folder = os.path.join(DYNAMIC_DATASET_PATH, gesture_name)
os.makedirs(gesture_folder, exist_ok=True)                                              # Ensure gesture folder exists

# Dataset collection parameters
print(f"Starting dataset collection for gesture: {gesture_name}")
print("Press 'q' to quit.")

sequence_length = 30                                                                    # Frames per sequence
sequences_to_collect = 30                                                               # Number of sequences per gesture
current_sequence = 0                                                                    # Counter for collected sequences
dynamic_sequence = []

# Frame rate control
target_fps = 10
frame_duration = 1.0 / target_fps
last_frame_time = time.time()


# Utility Functions
def mediapipe_detection(image, model):
    """Processes image through MediaPipe Pose model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                                      # Convert BGR to RGB
    image.flags.writeable = False                                                       # Optimize for processing
    results = model.process(image)                                                      # Pose estimation
    image.flags.writeable = True                                                        # Allow drawing
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                                      # Convert back to BGR
    return image, results


def draw_upper_body_landmarks(image, pose_results):
    """Draws only the upper body keypoints (head, shoulders, elbows, wrists, hips)."""
    if pose_results.pose_landmarks:
        upper_body_connections = [
            (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
            (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
        ]

        mp_drawing.draw_landmarks(
            image, pose_results.pose_landmarks, upper_body_connections,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),   # Bone color
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)   # Joint color
        )


def extract_upper_body_keypoints(pose_results):
    """Extracts only upper body keypoints (head, shoulders, elbows, wrists, hips)."""
    upper_body_indices = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
    ]

    if pose_results.pose_landmarks:
        keypoints = np.array([[pose_results.pose_landmarks.landmark[i].x,
                               pose_results.pose_landmarks.landmark[i].y,
                               pose_results.pose_landmarks.landmark[i].z]
                              for i in upper_body_indices]).flatten()
    else:
        keypoints = np.zeros(len(upper_body_indices) * 3)                               # Default if no pose detected

    return keypoints                                                                    # 13 keypoints * 3 values = 39 features


# Main Dataset Collection Loop
while True:
    current_time = time.time()
    if (current_time - last_frame_time) < frame_duration:
        continue                                                                        # Maintain the frame rate

    image = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_RGBA2RGB)

        # Process frame using MediaPipe Pose
        frame_rgb, pose_results = mediapipe_detection(frame, pose)

        # Extract only upper body keypoints
        keypoints = extract_upper_body_keypoints(pose_results)
        dynamic_sequence.append(keypoints)

        # Draw upper body pose landmarks
        draw_upper_body_landmarks(frame_rgb, pose_results)

        # Save sequence when enough frames are collected
        if len(dynamic_sequence) == sequence_length:
            # Generate unique filename
            file_index = len(os.listdir(gesture_folder))                               # Count existing files
            npy_path = os.path.join(gesture_folder, f"{file_index}.npy")

            np.save(npy_path, np.array(dynamic_sequence))
            print(f"[{current_sequence+1}/{sequences_to_collect}] Saved dynamic gesture sequence to {npy_path}")

            current_sequence += 1
            dynamic_sequence = []

            # Stop after collecting required sequences
            if current_sequence >= sequences_to_collect:
                print(f"Completed collection of {sequences_to_collect} sequences for {gesture_name}")
                break

        # Display Status
        cv2.putText(frame_rgb, f"Collecting Gesture: {gesture_name} ({current_sequence+1}/{sequences_to_collect})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Dynamic Gesture Dataset Collection", frame_rgb)

    last_frame_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close resources
cv2.destroyAllWindows()
zed.close()
print("Gesture dataset collection completed!")