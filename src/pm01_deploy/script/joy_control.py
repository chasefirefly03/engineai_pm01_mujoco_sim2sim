import pygame
import time
import sys
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

MAX_LINEAR_X = 0.5
MAX_ANGULAR_Z = 1.0

class JoyCmdVelPublisher(Node):
    def __init__(self):
        super().__init__('joy_cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.twist_msg = Twist()
        
    def timer_callback(self):
        self.publisher_.publish(self.twist_msg)

def main():
    rclpy.init()
    node = JoyCmdVelPublisher()
    
    # Initialize Pygame
    pygame.init()
    pygame.joystick.init()

    # Check for joysticks
    count = pygame.joystick.get_count()
    if count == 0:
        print("Error: No joystick found.")
        return

    # Use the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Initialized Joystick: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    print(f"Buttons: {joystick.get_numbuttons()}")
    print(f"Hats: {joystick.get_numhats()}")

    print("\nListening for events... Press Ctrl+C to stop.")

    try:
        while True:
            # Process event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button PRESSED: {event.button}")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Button RELEASED: {event.button}")
                elif event.type == pygame.JOYAXISMOTION:
                    # Axis 0: Left Stick X-axis (left/right) -> Turn speed (z)
                    if event.axis == 0:
                        node.twist_msg.angular.z = -event.value * MAX_ANGULAR_Z
                    # Axis 4: Right Stick Y-axis (up/down) -> Forward speed (x)
                    elif event.axis == 4:
                        node.twist_msg.linear.x = -event.value * MAX_LINEAR_X
                elif event.type == pygame.JOYHATMOTION:
                     print(f"Hat MOTION: {event.hat} Value={event.value}")

            # Process ROS 2 callbacks
            rclpy.spin_once(node, timeout_sec=0)

            # Sleep to reduce CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == "__main__":
    main()
