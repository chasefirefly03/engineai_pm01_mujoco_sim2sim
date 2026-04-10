import rclpy
from rclpy.node import Node
from interface_protocol.msg import JointCommand
import csv
import os
import math

# ================= 全局变量配置 =================

CONTROL_RATE = 500.0  # 控制频率 500Hz
CSV_FPS = 30.0        # CSV 动作原始帧率
SMOOTHING_ALPHA = 0.05 # 一阶低通滤波器系数 (0.0~1.0，值越小越平滑，1.0为无滤波直接输出)

JOINT_INDICES = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

JOINT_KP = [40.0, 40.0, 40.0, 40.0, 40.0, 
            40.0, 40.0, 40.0, 40.0, 40.0, 
            40.0]

JOINT_KD = [3.0, 3.0, 3.0, 3.0, 3.0, 
            3.0, 3.0, 3.0, 3.0, 3.0, 
            3.0]

DEFAULT_JOINT_POS = [0.0,  0.1,  0.0, 0.0, 0.0,
                     0.0, -0.1,  0.0, 0.0, 0.0,
                     0.0]

# ================================================

class CSVJointPublisher(Node):
    def __init__(self):
        super().__init__('csv_joint_publisher')
        self.publisher_ = self.create_publisher(JointCommand, '/hardware/joint_command', 10)
        
        # 按指定的控制频率创建定时器
        self.timer = self.create_timer(1.0 / CONTROL_RATE, self.timer_callback)
        
        # 运行时间记录，用于计算帧插值
        self.time_elapsed = 0.0
        
        # 记录平滑后的当前位置状态，作为滤波的初始值
        self.current_smooth_pos = [0.0] * 24
        
        # 读取CSV文件
        self.csv_data = []
        csv_path = os.path.join(os.path.dirname(__file__), 'sharkHand.csv')
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        self.csv_data.append([float(x) for x in row])
            self.get_logger().info(f'成功读取 {len(self.csv_data)} 行数据.')
        except Exception as e:
            self.get_logger().error(f'读取CSV文件失败: {e}')
            
        self.max_idx = len(self.csv_data)

        # 为不引起上电发生突变，对指定位置进行初始化赋予
        for i in range(min(len(DEFAULT_JOINT_POS), 24)):
            self.current_smooth_pos[i] = DEFAULT_JOINT_POS[i]
            
        if self.max_idx > 0:
            initial_row = self.csv_data[0]
            for i, j_idx in enumerate(JOINT_INDICES):
                if j_idx < 24 and i < len(initial_row):
                    self.current_smooth_pos[j_idx] = initial_row[i]

    def timer_callback(self):
        if self.max_idx == 0:
            return

        msg = JointCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        num_joints = 24
        
        # 将在此步计算纯理想目标的位置 (target_position)
        target_position = [0.0] * num_joints
        velocity = [0.0] * num_joints
        stiffness = [0.0] * num_joints
        damping = [0.0] * num_joints
        torque = [0.0] * num_joints
        feed_forward_torque = [0.0] * num_joints
        
        # ================== 1. 刚度与阻尼、默认位置赋初值 ==================
        for i in range(min(len(DEFAULT_JOINT_POS), num_joints)):
            target_position[i] = DEFAULT_JOINT_POS[i]
            
        for i, j_idx in enumerate(JOINT_INDICES):
            if j_idx < num_joints:
                stiffness[j_idx] = JOINT_KP[i]
                damping[j_idx] = JOINT_KD[i]
                
        # ================== 2. 帧插值计算 (线性) ==================
        # 将时间转化为原CSV频率下的帧索引
        float_idx = self.time_elapsed * CSV_FPS
        idx0 = int(float_idx) % self.max_idx
        idx1 = (idx0 + 1) % self.max_idx
        alpha_lerp = float_idx - int(float_idx)
        
        row0 = self.csv_data[idx0]
        row1 = self.csv_data[idx1]

        # 计算并更新关节目标位置
        for i, j_idx in enumerate(JOINT_INDICES):
            if j_idx < num_joints and i < len(row0) and i < len(row1):
                # 两帧之间的线性插值（LERP）
                interpolated_pos = (1.0 - alpha_lerp) * row0[i] + alpha_lerp * row1[i]
                target_position[j_idx] = interpolated_pos
                
        # ================== 3. 一阶低通滤波器实现完全平滑 ==================
        # LERP本身可能会在拐点（以及循环首尾衔接处）带来角速度的突变。
        # 利用指数平滑(LPF)彻底消除这种突变。
        for i in range(num_joints):
            self.current_smooth_pos[i] = (SMOOTHING_ALPHA * target_position[i] + 
                                         (1.0 - SMOOTHING_ALPHA) * self.current_smooth_pos[i])
                
        msg.position = self.current_smooth_pos.copy()
        msg.velocity = velocity
        msg.stiffness = stiffness
        msg.damping = damping
        msg.torque = torque
        msg.feed_forward_torque = feed_forward_torque

        self.publisher_.publish(msg)
        
        # 增加时间步进 (500Hz 对应推移 0.002s)
        self.time_elapsed += 1.0 / CONTROL_RATE

def main(args=None):
    rclpy.init(args=args)
    node = CSVJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
