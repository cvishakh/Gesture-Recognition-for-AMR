# Import libraries
import cv2
import csv
import os
import time
import numpy as np
import pyzed.sl as sl
import mediapipe as mp
import copy
import itertools

# Define gesture classes
GESTURE_CLASSES = {
    0: "closed_fist",
    1: "open_palm",
    2: "thumbs_up",
    3: "v_sign"
}

# Path for static dataset
STATIC_CSV_PATH = "dataset\static\keypoint.csv"

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

# Initialize MediaPipe for Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Determine collection mode
gesture_label = input("Enter the gesture label: ")

if int(gesture_label) not in GESTURE_CLASSES:
    print("Invalid gesture label! Please enter a number between 0 and 3.")
    exit(1)

print(f"Starting static dataset collection for gesture: {GESTURE_CLASSES[int(gesture_label)]}")
print("Press 's' to save static gesture, 'q' to quit.")

# Frame rate control
target_fps = 10
frame_duration = 1.0 / target_fps
last_frame_time = time.time()


def calc_landmark_list(image, landmarks):
    """Extracts (x, y) keypoints from MediaPipe landmarks"""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Extract x, y coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])                             # Only x, y

    return landmark_point


def pre_process_landmark(landmark_list):
    """Converts keypoints to relative coordinates and normalizes them"""
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]             # Wrist(index 0) position as reference
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    # Flatten the list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalize coordinate values
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value != 0:                                                             # Avoid division by zero
        temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list


# Main loop for dataset collection
while True:
    current_time = time.time()
    if (current_time - last_frame_time) < frame_duration:
        continue                                                                   # Maintain the frame rate

    image = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_RGBA2RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand Detection
        hand_results = hands.process(frame_rgb)

        # Draw Hand Landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract and Process Hand Keypoints
                hand_keypoints = calc_landmark_list(frame, hand_landmarks)
                processed_keypoints = pre_process_landmark(hand_keypoints)

                # Save static gesture keypoints
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    with open(STATIC_CSV_PATH, mode="a", newline="") as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([gesture_label, *processed_keypoints])
                        print(f"Saved static gesture for label: {gesture_label}")

        # Display Status
        cv2.putText(frame, f"Collecting static gesture for: {GESTURE_CLASSES[int(gesture_label)]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Static Gesture Dataset Collection", frame)

    last_frame_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close resources
cv2.destroyAllWindows()
zed.close()
print("Static gesture dataset collection completed!")