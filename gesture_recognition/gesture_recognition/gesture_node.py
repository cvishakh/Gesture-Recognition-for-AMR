#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import copy
import itertools

GESTURE_CLASSES = {
    0: "closed_fist",
    1: "open_palm",
    2: "thumbs_up",
    3: "v_sign",
    4: "pointing",
    5: "waving"
}

class GestureRecognitionNode(Node):
    def __init__(self):
        super().__init__('gesture_recognition_node')

        # Publishers
        self.gesture_pub = self.create_publisher(String, 'recognized_gesture', 10)
        self.point_pub = self.create_publisher(Point, 'pointing_coordinates', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, 10)

        self.bridge = CvBridge()
        self.depth_frame = None

        # Load models
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        self.static_model = tf.keras.models.load_model("/home/vishakh/ros2_ws/src/gesture_recognition/gesture_recognition/model/static_gesture_final.keras", compile=False)
        self.dynamic_model = tf.keras.models.load_model("/home/vishakh/ros2_ws/src/gesture_recognition/gesture_recognition/model/dynamic_gesture_final.keras", compile=False)

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        # Gesture tracking
        self.sequence_length = 30
        self.dynamic_sequence = []
        self.last_sequence = None
        self.no_movement_frames = 0
        self.gesture_active = None
        self.CONFIDENCE_THRESHOLD = 0.75
        self.MOVEMENT_THRESHOLD = 0.008

        self.get_logger().info("Gesture Recognition Node is running!")

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_frame = frame.copy()

        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        has_hand = hand_results.multi_hand_landmarks is not None
        pose_keypoints = self.extract_upper_body_keypoints(pose_results)
        movement = self.detect_movement(self.last_sequence, pose_keypoints)

        if movement:
            self.dynamic_sequence.append(pose_keypoints)
            self.last_sequence = pose_keypoints
            self.no_movement_frames = 0

            if len(self.dynamic_sequence) == self.sequence_length:
                input_data = np.expand_dims(self.dynamic_sequence, axis=0).astype('float32')
                prediction = self.dynamic_model.predict(input_data)
                pred_label = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > self.CONFIDENCE_THRESHOLD:
                    self.gesture_active = GESTURE_CLASSES[pred_label]
                    self.publish_gesture()
                self.dynamic_sequence = []

        elif has_hand:
            self.no_movement_frames += 1
            if self.no_movement_frames > 5:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    keypoints = self.calc_landmark_list(frame, hand_landmarks)
                    processed = self.pre_process_landmark(keypoints)
                    input_data = np.expand_dims(processed, axis=0).astype('float32')
                    prediction = self.static_model.predict(input_data)
                    pred_label = np.argmax(prediction)
                    confidence = np.max(prediction)

                    if confidence > self.CONFIDENCE_THRESHOLD:
                        self.gesture_active = GESTURE_CLASSES[pred_label]
                        self.publish_gesture()

                    self.mp_drawing.draw_landmarks(
                        self.display_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=6)
                    )

        if self.gesture_active == "pointing" and pose_results.pose_landmarks and self.depth_frame is not None:
            self.handle_pointing(pose_results, frame.shape)

        if pose_results.pose_landmarks:
            self.draw_upper_body_landmarks(self.display_frame, pose_results)

        if self.gesture_active:
            cv2.putText(self.display_frame, f"Gesture: {self.gesture_active}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Gesture Recognition using ZED2i", self.display_frame)
        cv2.waitKey(1)

    def draw_upper_body_landmarks(self, image, pose_results):
        upper_body_connections = [
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_EYE),
            (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_EYE),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP)
        ]

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            for start_idx, end_idx in upper_body_connections:
                start = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
                end = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
                cv2.line(image, start, end, (0, 255, 0), 4)

            for idx in [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_EYE, self.mp_pose.PoseLandmark.RIGHT_EYE,
                self.mp_pose.PoseLandmark.LEFT_EAR, self.mp_pose.PoseLandmark.RIGHT_EAR,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST,
                self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP
            ]:
                x, y = int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0])
                cv2.circle(image, (x, y), 6, (0, 0, 255), -1)

    def publish_gesture(self):
        self.gesture_pub.publish(String(data=self.gesture_active))
        self.get_logger().info(f"Published Gesture: {self.gesture_active}")

    def handle_pointing(self, pose_results, shape):
        h, w = shape[:2]
        rw = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        lw = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        pointing = rw if rw.visibility > lw.visibility else lw

        u, v = int(pointing.x * w), int(pointing.y * h)
        if 0 <= v < self.depth_frame.shape[0] and 0 <= u < self.depth_frame.shape[1]:
            z = self.depth_frame[v, u]
            if 0.1 < z < 5.0:
                point = Point()
                point.x = float(pointing.x * z)
                point.y = 0.0
                point.z = float(z)
                self.point_pub.publish(point)
                self.get_logger().info(f"Published pointing coordinate: Z = {z:.2f} m")

                # Draw point annotation and depth label for visualization
                cv2.circle(self.display_frame, (u, v), 8, (0, 255, 255), -1)
                cv2.putText(self.display_frame, f"Depth: {z:.2f}m", (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def calc_landmark_list(self, image, landmarks):
        h, w = image.shape[:2]
        return [[int(l.x * w), int(l.y * h)] for l in landmarks.landmark]

    def pre_process_landmark(self, landmark_list):
        temp = copy.deepcopy(landmark_list)
        base_x, base_y = temp[0]
        for i in range(len(temp)):
            temp[i][0] -= base_x
            temp[i][1] -= base_y
        flat = list(itertools.chain.from_iterable(temp))
        max_val = max(map(abs, flat)) or 1
        return [v / max_val for v in flat]

    def extract_upper_body_keypoints(self, pose_results):
        indices = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EYE, self.mp_pose.PoseLandmark.RIGHT_EYE,
            self.mp_pose.PoseLandmark.LEFT_EAR, self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP
        ]
        if pose_results.pose_landmarks:
            return np.array([[pose_results.pose_landmarks.landmark[i].x,
                              pose_results.pose_landmarks.landmark[i].y,
                              pose_results.pose_landmarks.landmark[i].z]
                             for i in indices]).flatten()
        else:
            return np.zeros(len(indices) * 3)

    def detect_movement(self, last, current):
        if last is None:
            return True
        diff = np.abs(np.array(current) - np.array(last))
        return np.mean(diff) > self.MOVEMENT_THRESHOLD

def main(args=None):
    rclpy.init(args=args)
    node = GestureRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()