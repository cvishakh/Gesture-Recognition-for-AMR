#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point
from enum import Enum

# Define different states of the robot
class RobotState(Enum):
    STANDBY = "standby"
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    MOVING_TO_POINT = "moving_to_point"
    LEARNING = "learning"

class GestureRobotController(Node):
    def __init__(self):
        super().__init__('gesture_robot_controller')

        # Subscribe to recognized gestures
        self.gesture_sub = self.create_subscription(
            String,
            'recognized_gesture',
            self.gesture_callback,
            10
        )

        # Subscribe to pointing coordinates
        self.point_sub = self.create_subscription(
            Point,
            'pointing_coordinates',
            self.point_callback,
            10
        )

        # Publisher for robot commands
        self.robot_command_pub = self.create_publisher(String, 'robot_command', 10)

        # Publisher for velocity control
        self.velocity_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Store last velocity before pausing (only for pause/resume)
        self.last_velocity = Twist()
        self.state = RobotState.STANDBY
        self.target_depth = None
        self.get_logger().info("Gesture Robot Controller Started in STANDBY mode.")

    def gesture_callback(self, msg):
        """Handles recognized gestures and manages state transitions."""
        gesture = msg.data
        command_msg = String()
        velocity_msg = Twist()

        if gesture == "waving":
            if self.state == RobotState.STANDBY:
                self.state = RobotState.WAITING
                command_msg.data = "User Login: Robot Ready"
            elif self.state == RobotState.WAITING:
                self.state = RobotState.STANDBY
                command_msg.data = "User Logout: Robot Standby"

        elif gesture == "thumbs_up":
            self.state = RobotState.WAITING
            command_msg.data = "Task Completed: Waiting for Next Task"
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0  # Stop motion

        elif gesture == "closed_fist":  # EMERGENCY STOP (DOES NOT REMEMBER LAST TASK)
            self.state = RobotState.STOPPED
            command_msg.data = "Emergency Stop: Task Cancelled"
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0  # Stop motion completely

        elif gesture == "open_palm":
            if self.state == RobotState.ACTIVE:
                self.state = RobotState.PAUSED
                command_msg.data = "Process Paused"
                self.last_velocity.linear.x = velocity_msg.linear.x
                self.last_velocity.angular.z = velocity_msg.angular.z
                velocity_msg.linear.x = 0.0
                velocity_msg.angular.z = 0.0  # Stop motion
            elif self.state == RobotState.PAUSED:
                self.state = RobotState.ACTIVE
                command_msg.data = "Process Resumed"
                velocity_msg = self.last_velocity  # Resume previous motion

        elif gesture == "pointing":
            self.state = RobotState.MOVING_TO_POINT
            command_msg.data = "Moving to Detected Depth Point"

        elif gesture == "v_sign":
            self.state = RobotState.LEARNING
            command_msg.data = "Learning Last Task"

        # Publish commands
        self.robot_command_pub.publish(command_msg)
        self.velocity_pub.publish(velocity_msg)

        self.get_logger().info(f"Detected Gesture: {gesture}, New State: {self.state.value}")

    def point_callback(self, msg):
        """Handles the depth value published by gesture_node when pointing."""
        if self.state != RobotState.MOVING_TO_POINT:
            return

        self.target_depth = msg.z
        self.get_logger().info(f"Received target depth: {self.target_depth:.2f} m")

        velocity_msg = Twist()

        # Only move if point is far enough
        if self.target_depth > 0.3:
            velocity_msg.linear.x = 0.2  # Move forward slowly
            self.velocity_pub.publish(velocity_msg)
        else:
            self.get_logger().info("Target reached or too close.")
            velocity_msg.linear.x = 0.0
            velocity_msg.angular.z = 0.0
            self.velocity_pub.publish(velocity_msg)

            # Switch to WAITING state
            self.state = RobotState.WAITING
            done_msg = String()
            done_msg.data = "Arrived at Pointed Location"
            self.robot_command_pub.publish(done_msg)
            self.get_logger().info("Arrived at pointed location. Switching to WAITING.")

def main(args=None):
    rclpy.init(args=args)
    node = GestureRobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()