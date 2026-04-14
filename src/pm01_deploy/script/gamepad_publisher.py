#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from interface_protocol.msg import GamepadKeys
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select
import pygame
import os

# ================= Configuration =================
# 可以在这里设置前进、旋转的速度 (Speed configurations)
FORWARD_SPEED = 0.6         # 前进速度 (m/s)
ROTATE_SPEED_LEFT = 0.5     # 左转速度 (rad/s)  (Button 1)
ROTATE_SPEED_RIGHT = -0.5   # 右转速度 (rad/s)  (Button 3)
# =================================================

class GamepadPublisher(Node):
    def __init__(self):
        super().__init__('gamepad_publisher')
        
        # 匹配 MessageHandler 中的 QoS
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.gamepad_pub_ = self.create_publisher(GamepadKeys, '/hardware/gamepad_keys', qos)
        
        # 匹配 CmdVel 的 Publisher (通常使用默认 QoS)
        self.cmd_vel_pub_ = self.create_publisher(Twist, '/pm01_cmd_vel', 10)
        
        self.get_logger().info('Gamepad Publisher Node Started')
        self.print_instructions()

        # 定时器以一定频率持续发布 Twist 指令
        self.cmd_vel_timer_ = self.create_timer(0.05, self.publish_twist_loop)
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0

    def print_instructions(self):
        print("\n=== Control Instructions ===")
        print("Keyboard Keys:")
        print("  z: Press Zero Torque")
        print("  s: Press START (Zero Torque -> Move to Default)")
        print("  a: Press A (Move to Default -> RL Control)")
        print("  d: Press SELECT/BACK (RL Control -> Damp)")
        print("  q: Quit")
        print("\nJoystick Buttons:")
        print("  Button 2: Forward")
        print("  Button 1: Rotate Left")
        print("  Button 3: Rotate Right")
        print("  Button 0: Zero Torque")
        print("  Button 4: START (Zero Torque -> Move to Default)")
        print("  Button 5: A (Move to Default -> RL Control)")
        print("============================\n")

    def publish_gamepad_key(self, key):
        msg = GamepadKeys()
        # 初始化数组
        msg.digital_states = [0] * 12
        msg.analog_states = [0.0] * 6
        
        if key == 's':
            msg.digital_states[7] = 1 # START
            self.get_logger().info('Publishing START command')
        elif key == 'a':
            msg.digital_states[2] = 1 # A
            self.get_logger().info('Publishing A command')
        elif key == 'd':
            msg.digital_states[6] = 1 # BACK/SELECT
            self.get_logger().info('Publishing BACK/SELECT command')
        elif key == 'z':
            msg.digital_states[0] = 1 # Zero Torque - 假设绑定到 digital_states[0]
            self.get_logger().info('Publishing Zero Torque command')
        else:
            return

        self.gamepad_pub_.publish(msg)

    def set_cmd_vel(self, linear_x, angular_z):
        self.current_linear_x = linear_x
        self.current_angular_z = angular_z

    def publish_twist_loop(self):
        msg = Twist()
        msg.linear.x = float(self.current_linear_x)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(self.current_angular_z)
        self.cmd_vel_pub_.publish(msg)

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    settings = termios.tcgetattr(sys.stdin)
    
    # 禁用 Pygame 的视频输出避免可能抛错
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # 初始化 Pygame 用于手柄输入读取
    pygame.init()
    pygame.joystick.init()

    # 检查手柄
    count = pygame.joystick.get_count()
    if count == 0:
        print("Warning: No joystick found. Operating in Keyboard-only mode.")
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Initialized Joystick: {joystick.get_name()}")

    rclpy.init(args=None)
    node = GamepadPublisher()
    
    # 手柄按键状态 (用于持续运动，而非仅点按一次)
    btn_2_pressed = False
    btn_1_pressed = False
    btn_3_pressed = False
    
    try:
        while rclpy.ok():
            # 1. 尝试读取键盘输入
            key = get_key(settings)
            if key == 'q':
                break
            if key in ['s', 'a', 'd', 'z']:
                node.publish_gamepad_key(key)
                
            # 2. 读取 Joystick 事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 2:
                        btn_2_pressed = True
                    elif event.button == 1:
                        btn_1_pressed = True
                    elif event.button == 3:
                        btn_3_pressed = True
                    elif event.button == 4:
                        node.publish_gamepad_key('s') # Button 4 -> START
                    elif event.button == 5:
                        node.publish_gamepad_key('a') # Button 5 -> A
                    elif event.button == 0:
                        node.publish_gamepad_key('z') # Button 0 -> Zero Torque (digital_states[0])
                    elif event.button == 6:
                        node.publish_gamepad_key('d') # 右回测
                elif event.type == pygame.JOYBUTTONUP:
                    if event.button == 2:
                        btn_2_pressed = False
                    elif event.button == 1:
                        btn_1_pressed = False
                    elif event.button == 3:
                        btn_3_pressed = False

            # 基于手柄按键计算当前速度
            vx = FORWARD_SPEED if btn_2_pressed else 0.0
            
            wz = 0.0
            if btn_1_pressed and not btn_3_pressed:
                wz = ROTATE_SPEED_LEFT
            elif btn_3_pressed and not btn_1_pressed:
                wz = ROTATE_SPEED_RIGHT
                
            node.set_cmd_vel(vx, wz)

            # Node 轮询
            rclpy.spin_once(node, timeout_sec=0.01)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # 发送停止指令
        node.set_cmd_vel(0.0, 0.0)
        node.publish_twist_loop()
        
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == '__main__':
    main()
