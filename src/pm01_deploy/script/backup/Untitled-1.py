#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from interface_protocol.msg import GamepadKeys
import sys
import termios
import tty
import select

class GamepadPublisher(Node):
    def __init__(self):
        super().__init__('gamepad_publisher')
        # Matching QoS: BEST_EFFORT, VOLATILE to match the subscriber in MessageHandler
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.publisher_ = self.create_publisher(GamepadKeys, '/hardware/gamepad_keys', qos)
        self.get_logger().info('Gamepad Publisher Node Started')
        self.print_instructions()

    def print_instructions(self):
        print("\nControl Keys:")
        print("  s: Press START (Zero Torque -> Move to Default)")
        print("  a: Press A (Move to Default -> RL Control)")
        print("  d: Press SELECT/BACK (RL Control -> Damp)")
        print("  q: Quit")

    def publish_key(self, key):
        msg = GamepadKeys()
        # Initialize arrays
        msg.digital_states = [0] * 12
        msg.analog_states = [0.0] * 6
        
        # Mapping based on common gamepads and header file usually found in msg
        # START = 7
        # A = 2
        # BACK = 6
        
        if key == 's':
            msg.digital_states[7] = 1 # START
            self.get_logger().info('Publishing START command')
        elif key == 'a':
            msg.digital_states[2] = 1 # A
            self.get_logger().info('Publishing A command')
        elif key == 'd':
            msg.digital_states[6] = 1 # BACK/SELECT
            self.get_logger().info('Publishing BACK/SELECT command')
        else:
            return

        self.publisher_.publish(msg)

def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init(args=None)
    node = GamepadPublisher()
    
    try:
        while rclpy.ok():
            key = get_key()
            if key == 'q':
                break
            if key in ['s', 'a', 'd']:
                node.publish_key(key)
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception as e:
        print(e)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # Restore terminal settings
        if 'settings' in locals():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
