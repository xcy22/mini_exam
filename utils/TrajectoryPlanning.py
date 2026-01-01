import numpy as np

PI = np.pi
FRAC_PI_2_3 = np.pi/3*2

class Trajectory:
    def __init__(self,T,q_s,q_e):
        self.T=T
        self.q_s=q_s
        self.q_e=q_e
        self.q_min=[-PI,-FRAC_PI_2_3,-FRAC_PI_2_3,-PI,-FRAC_PI_2_3,-PI,-PI]
        self.q_max=[PI,FRAC_PI_2_3,FRAC_PI_2_3,PI,FRAC_PI_2_3,PI,PI]
        self.v_min=[-PI]*6
        self.v_max=[PI]*6

    def tra_plan(self):
        """
        三次多项式轨迹规划
        轨迹方程：q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        边界条件：q(0)=q_s, q(T)=q_e, v(0)=0, v(T)=0
        
        参数：
        T: 轨迹时间，默认为1.0秒
        
        返回：
        系数矩阵A，形状为(n_joints, 4)，每一行对应一个关节的[a0, a1, a2, a3]
        """
        T=self.T
        # 确定关节数（根据q_s的长度）
        n_joints = len(self.q_s)
        
        # 初始化系数矩阵，每行对应一个关节的4个系数
        A = np.zeros((n_joints, 4))
        
        # 对于每个关节，计算三次多项式系数
        for i in range(n_joints):
            q0 = self.q_s[i]  # 初始位置
            qf = self.q_e[i]  # 最终位置
            a0 = q0
            a1 = 0
            
            # 解方程得到a2和a3
            # 从条件3和4得到：
            # a2*T^2 + a3*T^3 = qf - q0
            # 2*a2*T + 3*a3*T^2 = 0
            
            # 写成矩阵形式求解
            # [T^2, T^3; 2T, 3T^2] * [a2; a3] = [qf - q0; 0]
            
            T_sq = T**2
            T_cu = T**3
            
            # 构建系数矩阵
            M = np.array([[T_sq, T_cu],
                          [2*T, 3*T_sq]])
            
            # 构建右侧向量
            b = np.array([qf - q0, 0])
            
            # 求解线性方程组
            a2_a3 = np.linalg.solve(M, b)
            
            a2 = a2_a3[0]
            a3 = a2_a3[1]
            
            # 将系数存储到矩阵中
            A[i, 0] = a0
            A[i, 1] = a1
            A[i, 2] = a2
            A[i, 3] = a3
        
        self.A = A
        
        # 可选：验证轨迹是否满足约束
        self._validate_trajectory(T)
        print(A)
        return A
    
    def _validate_trajectory(self, T, n_points=100):
        """
        验证轨迹是否满足关节位置和速度约束
        
        参数：
        T: 轨迹时间
        n_points: 采样点数量
        """
        # 采样时间点
        time_points = np.linspace(0, T, n_points)
        
        for i in range(len(self.q_s)):
            # 计算每个采样点的位置和速度
            a0, a1, a2, a3 = self.A[i]
            
            # 位置：q(t) = a0 + a1*t + a2*t^2 + a3*t^3
            pos = a0 + a1*time_points + a2*time_points**2 + a3*time_points**3
            
            # 速度：v(t) = a1 + 2*a2*t + 3*a3*t^2
            vel = a1 + 2*a2*time_points + 3*a3*time_points**2
            
            # 检查位置约束
            pos_min, pos_max = self.q_min[i], self.q_max[i]
            if np.any(pos < pos_min) or np.any(pos > pos_max):
                print(f"警告: 关节{i+1}的位置超出约束范围 [{pos_min:.3f}, {pos_max:.3f}]")
            
            # 检查速度约束（如果存在速度约束）
            if i < len(self.v_min):
                vel_min, vel_max = self.v_min[i], self.v_max[i]
                if np.any(vel < vel_min) or np.any(vel > vel_max):
                    print(f"警告: 关节{i+1}的速度超出约束范围 [{vel_min:.3f}, {vel_max:.3f}]")
    
    def evaluate(self, t):
        """
        在时间t处评估轨迹
        
        参数：
        t: 时间点（0 ≤ t ≤ T）
        T: 轨迹时间
        
        返回：
        q: 关节位置
        v: 关节速度
        """
        T=self.T
        if not hasattr(self, 'A'):
            raise ValueError("请先调用tra_plan方法生成轨迹")
        
        n_joints = len(self.q_s)
        q = np.zeros(n_joints)
        v = np.zeros(n_joints)
        
        for i in range(n_joints):
            a0, a1, a2, a3 = self.A[i]
            # 位置
            q[i] = a0 + a1*t + a2*t**2 + a3*t**3
            # 速度
            v[i] = a1 + 2*a2*t + 3*a3*t**2
        
        return q, v

