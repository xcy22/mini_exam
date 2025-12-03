import numpy as np
import math

class InverseKinematicsSolver:
    def __init__(self, dh_params, max_iter=100, tol=1e-4, learning_rate=1.0):
        """
        初始化 IK 求解器
        :param dh_params: DH 参数列表，格式为 [[a, alpha, d, theta_offset], ...]
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值 (位置和姿态误差)
        :param learning_rate: 学习率 (步长)
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)
        self.max_iter = max_iter
        self.tol = tol
        self.lr = learning_rate

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

    def _compute_jacobian(self, transforms, end_effector_pos):
        """
        计算几何雅可比矩阵 (6xN)
        """
        J = np.zeros((6, self.num_joints))
        
        # 基座坐标系原点和Z轴
        prev_pos = np.array([0, 0, 0])
        prev_z = np.array([0, 0, 1])
        
        for i in range(self.num_joints):
            # 当前关节的 Z 轴方向 (旋转轴)
            z_axis = prev_z
            # 当前关节的位置
            p_axis = prev_pos
            
            # 位置部分 (v = w x r)
            # r = vector from joint i to end effector
            r = end_effector_pos - p_axis
            J[:3, i] = np.cross(z_axis, r)
            
            # 姿态部分 (w = z)
            J[3:, i] = z_axis
            
            # 更新为下一关节的参考系 (从 transforms 列表中获取)
            T = transforms[i]
            prev_pos = T[:3, 3]
            prev_z = T[:3, 2]
            
        return J

    def _rotation_error(self, R_target, R_current):
        """
        计算姿态误差 (旋转向量形式)
        Error = 0.5 * (n x n_d + s x s_d + a x a_d)
        """
        n_cur = R_current[:, 0]
        s_cur = R_current[:, 1]
        a_cur = R_current[:, 2]
        
        n_des = R_target[:, 0]
        s_des = R_target[:, 1]
        a_des = R_target[:, 2]
        
        err = 0.5 * (np.cross(n_cur, n_des) + np.cross(s_cur, s_des) + np.cross(a_cur, a_des))
        return err

    def solve(self, target_pos, target_rot, initial_joints=None):
        """
        迭代求解逆运动学
        :param target_pos: 目标位置 [x, y, z]
        :param target_rot: 目标旋转矩阵 (3x3)
        :param initial_joints: 初始关节角度猜测 (可选)
        :return: (success, joint_angles)
        """
        if initial_joints is None:
            q = np.zeros(self.num_joints)
        else:
            q = np.array(initial_joints, dtype=float)
            
        target_pos = np.array(target_pos)
        
        for iteration in range(self.max_iter):
            # 1. 计算当前正运动学
            curr_pos, curr_rot, transforms = self.forward_kinematics(q)
            
            # 2. 计算误差向量 (6x1)
            pos_err = target_pos - curr_pos
            rot_err = self._rotation_error(target_rot, curr_rot)
            error = np.concatenate((pos_err, rot_err))
            
            # 3. 检查是否收敛
            if np.linalg.norm(error) < self.tol:
                # 将角度归一化到 [-pi, pi]
                q = (q + np.pi) % (2 * np.pi) - np.pi
                return True, q
            
            # 4. 计算雅可比矩阵
            J = self._compute_jacobian(transforms, curr_pos)
            
            # 5. 计算雅可比伪逆 (Damped Least Squares / Levenberg-Marquardt 简化版)
            # J_pinv = (J.T * J + lambda * I)^-1 * J.T
            lambda_factor = 0.01
            J_pinv = np.dot(np.linalg.inv(np.dot(J.T, J) + lambda_factor * np.eye(self.num_joints)), J.T)
            
            # 6. 更新关节角度: q_new = q_old + alpha * J_pinv * error
            delta_q = np.dot(J_pinv, error)
            q += self.lr * delta_q
            
        return False, q

# --- 测试代码 ---
if __name__ == "__main__":
    # 示例：Jaka Mini 2 的近似 DH 参数 (a, alpha, d, theta_offset)
    # 注意：这里需要填入真实的 DH 参数才能准确工作
    # 格式: [a, alpha, d, offset]
    jaka_dh = [
        [0,       np.pi/2,  0.120, 0],
        [-0.425,  0,        0,     0],
        [-0.395,  0,        0,     0],
        [0,       np.pi/2,  0.090, 0],
        [0,      -np.pi/2,  0.095, 0],
        [0,       0,        0.070, 0]
    ]

    ik_solver = InverseKinematicsSolver(jaka_dh)

    # 设定一个目标 (假设这是某个已知点的位姿)
    target_position = [0.4, 0.1, 0.3]
    target_rotation = np.eye(3) # 单位阵，即无旋转

    print(f"Target Pos: {target_position}")
    
    success, joints = ik_solver.solve(target_position, target_rotation)
    
    if success:
        print("IK Solved Successfully!")
        print("Joint Angles (rad):", np.round(joints, 4))
        
        # 验证结果
        final_pos, _, _ = ik_solver.forward_kinematics(joints)
        print("Final Pos:", np.round(final_pos, 4))
        print("Error:", np.linalg.norm(target_position - final_pos))
    else:
        print("IK Failed to Converge within iterations.")