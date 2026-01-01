import numpy as np
import math

class ForwardKinematicsSolver:
    def __init__(self, dh_params):
        """
        初始化 FK 求解器
        :param dh_params: DH 参数列表，格式为 [[a, alpha, d, theta_offset], ...]
        """
        self.dh_params = dh_params

    def _dh_transform(self, a, alpha, d, theta):
        """计算单个关节的变换矩阵 (标准 DH)"""
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])

    def forward_kinematics(self, joints):
        """
        正运动学：计算末端位姿
        :param joints: 关节角度列表 (弧度)
        :return: (position, rotation_matrix, all_transforms)
        """
        T = np.eye(4)
        transforms = []
        
        for i, params in enumerate(self.dh_params):
            a, alpha, d, offset = params
            theta = joints[i] + offset
            T_i = self._dh_transform(a, alpha, d, theta)
            T = np.dot(T, T_i)
            transforms.append(T)
            
        return T[:3, 3], T[:3, :3], transforms

if __name__ == "__main__":
    # 示例：Jaka Mini 2 的 DH 参数
    jaka_dh = [
        [0,       np.pi/2,  0.187, 0],
        [0.210,   0,        0,     np.pi/2],
        [0,  np.pi/2,        0,     np.pi/2],
        [0,       np.pi/2,  0.2105, np.pi],
        [0,      np.pi/2,  0.006, np.pi],
        [0,       0,        0.1593, 0]
    ]

    fk_solver = ForwardKinematicsSolver(jaka_dh)
    
    # 测试关节角 (全零)
    test_joints = [-30.928, 27.715, 41.188, -85.189, 59.993, -140.573]
    # 注意：如果输入是角度，需要先转换为弧度
    test_joints = np.radians(test_joints) 

    pos, rot, _ = fk_solver.forward_kinematics(test_joints)
    
    print("Joints:", test_joints)
    print("Local Position:", np.round(pos, 4))
    print("Local Rotation Matrix:\n", np.round(rot, 4))

    # --- 计算绝对坐标系下的位姿 ---
    
    # 基座参数
    Tbase_master = np.array([[1.0,0.0,0.0,0.275],[0.0,1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    Tbase_slave = np.array([[-1.0,0.0,0.0,-0.275],[0.0,-1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])

    # 构建局部齐次变换矩阵
    T_local = np.eye(4)
    T_local[:3, :3] = rot
    T_local[:3, 3] = pos

    # 1. 主臂绝对位姿
    T_abs_master = np.dot(Tbase_master, T_local)
    pos_abs_master = T_abs_master[:3, 3]
    rot_abs_master = T_abs_master[:3, :3]

    print("\n[Master Arm Absolute Pose]")
    print("Position:", np.round(pos_abs_master, 4))
    print("Rotation Matrix:\n", np.round(rot_abs_master, 4))

    # 2. 从臂绝对位姿 (假设使用相同的关节角进行演示)
    T_abs_slave = np.dot(Tbase_slave, T_local)
    pos_abs_slave = T_abs_slave[:3, 3]
    rot_abs_slave = T_abs_slave[:3, :3]

    print("\n[Slave Arm Absolute Pose]")
    print("Position:", np.round(pos_abs_slave, 4))
    print("Rotation Matrix:\n", np.round(rot_abs_slave, 4))