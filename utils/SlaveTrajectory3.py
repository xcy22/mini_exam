import numpy as np
import json
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import math

PI = np.pi
FRAC_PI_2_3 = np.pi/3*2

class SlaveTrajectoryOptimizer:
    """
    从臂轨迹规划器 - 使用SLSQP优化算法
    在满足关节限位、速度限位和末端距离约束的条件下规划平滑轨迹
    """
    
    def __init__(self, T, q_s, q_e,Tbase_master,Tbase_slave,master_trajectory_file='trajectory_data.json', 
                 d_min=0.2, d_max=0.3, dt=1/125):
        """
        初始化从臂轨迹规划器
        
        参数:
        T: 轨迹总时间 (秒)
        q_s: 从臂起始关节位置 (6个关节)
        q_e: 从臂结束关节位置 (6个关节)
        master_trajectory_file: 主臂轨迹数据文件路径
        d_min: 末端最小距离约束 (米)
        d_max: 末端最大距离约束 (米)
        Tbase=(4, dtype=np.float64)
        dt: 时间步长 (秒)
        """
        self.dt = dt
        self.T = T
        self.q_s = np.array(q_s, dtype=np.float64)
        self.q_e = np.array(q_e, dtype=np.float64)
        self.d_min = d_min
        self.d_max = d_max
        self.Tbase_master=Tbase_master
        self.Tbase_slave=Tbase_slave
        # 关节约束 (6个关节)
        self.q_min = np.array([-PI, -FRAC_PI_2_3, -FRAC_PI_2_3, -PI, -FRAC_PI_2_3, -PI], dtype=np.float64)
        self.q_max = np.array([PI, FRAC_PI_2_3, FRAC_PI_2_3, PI, FRAC_PI_2_3, PI], dtype=np.float64)
        self.v_min = np.array([-PI] * 6, dtype=np.float64)
        self.v_max = np.array([PI] * 6, dtype=np.float64)
        
        # 加载主臂轨迹
        with open(master_trajectory_file, 'r') as f:
            self.master_trajectory = json.load(f)
        
        # 机械臂DH参数 (示例：6自由度机械臂)
        self.dh_params = [
            {'a': 0.0,   'd': 0.187, 'alpha': PI/2, 'theta':0},          # 关节1
            {'a': 0.210, 'd': 0.0,   'alpha': 0,    'theta':PI/2},      # 关节2
            {'a': 0.0,   'd': 0.0,   'alpha': PI/2, 'theta':PI/2},          # 关节3
            {'a': 0.0,   'd': 0.2105,'alpha': PI/2, 'theta':PI},      # 关节4
            {'a': 0.0,   'd': 0.006, 'alpha': PI/2, 'theta':PI},       # 关节5
            {'a': 0.0,   'd': 0.1593,'alpha': 0,    'theta':0},      # 关节6
        ]
        
        # 时间点和索引映射
        self.time_points = np.arange(0, T + dt, dt)
        self.n_points = len(self.time_points)
        
        # 主臂末端位置缓存
        self.master_positions = self._compute_master_positions()
        
        # 优化变量：每个关节的三次多项式系数 [a0, a1, a2, a3]
        self.n_joints = 6
        self.n_coeffs_per_joint = 4
        self.n_total_coeffs = self.n_joints * self.n_coeffs_per_joint
        
        # 优化结果
        self.optimized_coeffs = None
        self.optimization_result = None
    
    def _compute_master_positions(self):
        """预计算主臂末端位置"""
        master_positions = []
        
        for i in range(min(self.n_points, len(self.master_trajectory))):
            q_master = np.array(self.master_trajectory[i]['Joint'], dtype=np.float64)
            master_positions.append(self.forward_kinematics(q_master,self.Tbase_master))
        
        # 如果点数不够，用最后一个位置填充
        while len(master_positions) < self.n_points:
            master_positions.append(master_positions[-1])
        
        return np.array(master_positions)
    
    def forward_kinematics(self, q, Tbase):
        """
        正向运动学 - 计算机械臂末端位置
        
        参数:
        q: 关节角度 (6个)
        
        返回:
        position: 末端位置 [x, y, z]
        """
        T = Tbase
        
        for i in range(len(q)):
            theta = q[i] + self.dh_params[i]['theta']
            d = self.dh_params[i]['d']
            a = self.dh_params[i]['a']
            alpha = self.dh_params[i]['alpha']
            
            # DH变换矩阵
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            cos_alpha = np.cos(alpha)
            sin_alpha = np.sin(alpha)
            
            Ti = np.array([
                [cos_theta, -sin_theta*cos_alpha, sin_theta*sin_alpha, a*cos_theta],
                [sin_theta, cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta],
                [0, sin_alpha, cos_alpha, d],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            
            T = T @ Ti
        
        # 返回末端位置
        return T[:3, 3]
    
    def _get_initial_guess(self):
        """
        生成初始猜测：使用标准三次多项式满足起点终点条件
        """
        coeffs = np.zeros(self.n_total_coeffs, dtype=np.float64)
        
        for i in range(self.n_joints):
            q0 = self.q_s[i]
            qf = self.q_e[i]
            T = self.T
            
            # 标准三次多项式系数
            a0 = q0
            a1 = 0.0
            a2 = (3.0 * (qf - q0)) / (T**2)
            a3 = (-2.0 * (qf - q0)) / (T**3)
            
            # 存储系数
            idx = i * self.n_coeffs_per_joint
            coeffs[idx:idx+self.n_coeffs_per_joint] = [a0, a1, a2, a3]
        
        return coeffs
    
    def _coeffs_to_matrix(self, coeffs):
        """将一维系数数组转换为矩阵形式"""
        return coeffs.reshape(self.n_joints, self.n_coeffs_per_joint)
    
    def compute_position(self, coeffs, t):
        """根据系数计算时间t的位置"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        positions = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            a0, a1, a2, a3 = coeffs_matrix[i]
            positions[i] = a0 + a1*t + a2*(t**2) + a3*(t**3)
        
        return positions
    
    def compute_velocity(self, coeffs, t):
        """根据系数计算时间t的速度"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        velocities = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            a0, a1, a2, a3 = coeffs_matrix[i]
            velocities[i] = a1 + 2*a2*t + 3*a3*(t**2)
        
        return velocities
    
    def compute_acceleration(self, coeffs, t):
        """根据系数计算时间t的加速度"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        accelerations = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            a0, a1, a2, a3 = coeffs_matrix[i]
            accelerations[i] = 2*a2 + 6*a3*t
        
        return accelerations
    
    def objective_function(self, coeffs):
        """
        目标函数：最小化加速度平方和，使轨迹平滑
        
        参数:
        coeffs: 优化变量 (所有关节的系数)
        
        返回:
        目标函数值
        """
        # 采样时间点
        n_samples = 50
        time_samples = np.linspace(0, self.T, n_samples)
        
        total_acceleration_squared = 0.0
        
        for t in time_samples:
            accelerations = self.compute_acceleration(coeffs, t)
            total_acceleration_squared += np.sum(accelerations**2)
        
        return total_acceleration_squared
    
    def _eq_constraint_start_position(self, coeffs):
        """等式约束：起点位置"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        constraints = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            a0 = coeffs_matrix[i, 0]  # a0是t=0时的位置
            constraints[i] = a0 - self.q_s[i]
        
        return constraints
    
    def _eq_constraint_end_position(self, coeffs):
        """等式约束：终点位置"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        constraints = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            a0, a1, a2, a3 = coeffs_matrix[i]
            q_end = a0 + a1*self.T + a2*(self.T**2) + a3*(self.T**3)
            constraints[i] = q_end - self.q_e[i]
        
        return constraints
    
    def _eq_constraint_start_velocity(self, coeffs):
        """等式约束：起点速度"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        constraints = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            a1 = coeffs_matrix[i, 1]  # a1是t=0时的速度
            constraints[i] = a1 - 0.0  # v(0) = 0
        
        return constraints
    
    def _eq_constraint_end_velocity(self, coeffs):
        """等式约束：终点速度"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        constraints = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            a1, a2, a3 = coeffs_matrix[i, 1:4]
            v_end = a1 + 2*a2*self.T + 3*a3*(self.T**2)
            constraints[i] = v_end - 0.0  # v(T) = 0
        
        return constraints
    
    def _ineq_constraint_joint_position(self, coeffs):
        """
        不等式约束：关节位置限制
        返回：所有采样点的位置与限位的差值
        """
        # 采样时间点
        n_samples = 20
        time_samples = np.linspace(0, self.T, n_samples)
        
        constraints = []
        
        for t in time_samples:
            positions = self.compute_position(coeffs, t)
            
            for i in range(self.n_joints):
                # q_min ≤ q ≤ q_max 转化为两个不等式：
                # 1) q - q_min ≥ 0
                # 2) q_max - q ≥ 0
                constraints.append(positions[i] - self.q_min[i])  # q - q_min ≥ 0
                constraints.append(self.q_max[i] - positions[i])  # q_max - q ≥ 0
        
        return np.array(constraints, dtype=np.float64)
    
    def _ineq_constraint_joint_velocity(self, coeffs):
        """
        不等式约束：关节速度限制
        """
        n_samples = 20
        time_samples = np.linspace(0, self.T, n_samples)
        
        constraints = []
        
        for t in time_samples:
            velocities = self.compute_velocity(coeffs, t)
            
            for i in range(self.n_joints):
                # v_min ≤ v ≤ v_max 转化为两个不等式：
                constraints.append(velocities[i] - self.v_min[i])  # v - v_min ≥ 0
                constraints.append(self.v_max[i] - velocities[i])  # v_max - v ≥ 0
        
        return np.array(constraints, dtype=np.float64)
    
    def _ineq_constraint_end_distance(self, coeffs):
        """
        不等式约束：末端距离限制
        d_min ≤ distance ≤ d_max
        """
        # 采样时间点
        n_samples = 15
        time_indices = np.linspace(0, self.n_points-1, n_samples, dtype=int)
        
        constraints = []
        
        for idx in time_indices:
            t = idx * self.dt
            q_slave = self.compute_position(coeffs, t)
            
            # 计算从臂末端位置
            slave_pos = self.forward_kinematics(q_slave,self.Tbase_slave)
            
            # 获取主臂末端位置
            if idx < len(self.master_positions):
                master_pos = self.master_positions[idx]
            else:
                master_pos = self.master_positions[-1]
            
            # 计算距离
            distance = np.linalg.norm(slave_pos - master_pos)
            
            # d_min ≤ distance ≤ d_max 转化为两个不等式：
            # 1) distance - d_min ≥ 0
            # 2) d_max - distance ≥ 0
            constraints.append(distance - self.d_min)  # distance - d_min ≥ 0
            constraints.append(self.d_max - distance)  # d_max - distance ≥ 0
        
        return np.array(constraints, dtype=np.float64)
    
    def optimize(self, max_iter=1000, ftol=1e-6, disp=True):
        """
        执行轨迹优化
        
        参数:
        max_iter: 最大迭代次数
        ftol: 函数值容忍度
        disp: 是否显示优化过程信息
        
        返回:
        success: 优化是否成功
        """
        print("开始从臂轨迹优化...")
        print(f"优化变量数: {self.n_total_coeffs}")
        print(f"时间范围: 0.0 到 {self.T} 秒")
        print(f"距离约束: {self.d_min} 到 {self.d_max} 米")
        
        # 初始猜测
        initial_coeffs = self._get_initial_guess()
        
        # 定义变量边界
        bounds = Bounds(lb=-100.0, ub=100.0, keep_feasible=True)
        
        # 定义约束
        constraints = [
            # 等式约束
            {'type': 'eq', 'fun': self._eq_constraint_start_position},
            {'type': 'eq', 'fun': self._eq_constraint_end_position},
            {'type': 'eq', 'fun': self._eq_constraint_start_velocity},
            {'type': 'eq', 'fun': self._eq_constraint_end_velocity},
            
            # 不等式约束
            {'type': 'ineq', 'fun': self._ineq_constraint_joint_position},
            {'type': 'ineq', 'fun': self._ineq_constraint_joint_velocity},
            {'type': 'ineq', 'fun': self._ineq_constraint_end_distance},
        ]
        
        # 优化选项
        options = {
            'maxiter': max_iter,
            'ftol': ftol,
            'disp': disp,
            'iprint': 1,
        }
        
        # 执行优化
        self.optimization_result = minimize(
            self.objective_function,
            initial_coeffs,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        
        # 保存优化结果
        self.optimized_coeffs = self.optimization_result.x
        
        # 输出优化结果
        print(f"\n优化完成:")
        print(f"  成功: {self.optimization_result.success}")
        print(f"  消息: {self.optimization_result.message}")
        print(f"  目标函数值: {self.optimization_result.fun:.6f}")
        print(f"  迭代次数: {self.optimization_result.nit}")
        
        return self.optimization_result.success
    
    def validate_constraints(self, tolerance=1e-4):
        """
        验证优化结果是否满足所有约束
        
        参数:
        tolerance: 允许的误差范围
        
        返回:
        all_satisfied: 是否所有约束都满足
        """
        if self.optimized_coeffs is None:
            raise ValueError("请先执行optimize()方法进行优化")
        
        print("\n验证约束条件:")
        
        coeffs_matrix = self._coeffs_to_matrix(self.optimized_coeffs)
        all_satisfied = True
        
        # 验证等式约束
        print("1. 等式约束验证:")
        
        # 起点位置
        for i in range(self.n_joints):
            a0 = coeffs_matrix[i, 0]
            error = abs(a0 - self.q_s[i])
            if error > tolerance:
                print(f"   关节{i+1}: 起点位置误差 {error:.6f} > {tolerance}")
                all_satisfied = False
        
        # 终点位置
        for i in range(self.n_joints):
            a0, a1, a2, a3 = coeffs_matrix[i]
            q_end = a0 + a1*self.T + a2*(self.T**2) + a3*(self.T**3)
            error = abs(q_end - self.q_e[i])
            if error > tolerance:
                print(f"   关节{i+1}: 终点位置误差 {error:.6f} > {tolerance}")
                all_satisfied = False
        
        # 起点速度
        for i in range(self.n_joints):
            a1 = coeffs_matrix[i, 1]
            error = abs(a1 - 0.0)
            if error > tolerance:
                print(f"   关节{i+1}: 起点速度误差 {error:.6f} > {tolerance}")
                all_satisfied = False
        
        # 终点速度
        for i in range(self.n_joints):
            a1, a2, a3 = coeffs_matrix[i, 1:4]
            v_end = a1 + 2*a2*self.T + 3*a3*(self.T**2)
            error = abs(v_end - 0.0)
            if error > tolerance:
                print(f"   关节{i+1}: 终点速度误差 {error:.6f} > {tolerance}")
                all_satisfied = False
        
        # 验证不等式约束
        print("\n2. 不等式约束验证:")
        
        # 采样检查关节位置约束
        n_check = 10
        time_check = np.linspace(0, self.T, n_check)
        
        for t in time_check:
            positions = self.compute_position(self.optimized_coeffs, t)
            velocities = self.compute_velocity(self.optimized_coeffs, t)
            
            for i in range(self.n_joints):
                # 检查位置约束
                if positions[i] < self.q_min[i] - tolerance:
                    print(f"   时间{t:.2f}s, 关节{i+1}: 位置 {positions[i]:.4f} < 下限 {self.q_min[i]:.4f}")
                    all_satisfied = False
                elif positions[i] > self.q_max[i] + tolerance:
                    print(f"   时间{t:.2f}s, 关节{i+1}: 位置 {positions[i]:.4f} > 上限 {self.q_max[i]:.4f}")
                    all_satisfied = False
                
                # 检查速度约束
                if velocities[i] < self.v_min[i] - tolerance:
                    print(f"   时间{t:.2f}s, 关节{i+1}: 速度 {velocities[i]:.4f} < 下限 {self.v_min[i]:.4f}")
                    all_satisfied = False
                elif velocities[i] > self.v_max[i] + tolerance:
                    print(f"   时间{t:.2f}s, 关节{i+1}: 速度 {velocities[i]:.4f} > 上限 {self.v_max[i]:.4f}")
                    all_satisfied = False
        
        # 验证末端距离约束
        print("\n3. 末端距离约束验证:")
        n_check = 10
        time_indices = np.linspace(0, self.n_points-1, n_check, dtype=int)
        
        for idx in time_indices:
            t = idx * self.dt
            q_slave = self.compute_position(self.optimized_coeffs, t)
            slave_pos = self.forward_kinematics(q_slave,self.Tbase_slave)
            
            if idx < len(self.master_positions):
                master_pos = self.master_positions[idx]
            else:
                master_pos = self.master_positions[-1]
            
            distance = np.linalg.norm(slave_pos - master_pos)
            
            if distance < self.d_min - tolerance:
                print(f"   时间{t:.2f}s: 距离 {distance:.4f} < 下限 {self.d_min:.4f}")
                all_satisfied = False
            elif distance > self.d_max + tolerance:
                print(f"   时间{t:.2f}s: 距离 {distance:.4f} > 上限 {self.d_max:.4f}")
                all_satisfied = False
            else:
                print(f"   时间{t:.2f}s: 距离 {distance:.4f} 在范围内 [{self.d_min:.4f}, {self.d_max:.4f}]")
        
        if all_satisfied:
            print("\n所有约束条件满足!")
        else:
            print("\n警告: 部分约束条件不满足!")
        
        return all_satisfied
    
    def generate_trajectory(self):
        """
        生成轨迹数据
        
        返回:
        trajectory: 轨迹数据列表
        """
        if self.optimized_coeffs is None:
            raise ValueError("请先执行optimize()方法进行优化")
        
        trajectory = []
        
        t = 0.0
        while t <= self.T + 1e-10:  # 加上小量避免浮点误差
            q = self.compute_position(self.optimized_coeffs, t)
            
            joint_data = {
                "Joint": q.tolist()
            }
            trajectory.append(joint_data)
            
            t += self.dt
        
        return trajectory
    
    def save_trajectory(self, filename='slave_trajectory_data.json'):
        """
        保存轨迹到JSON文件
        
        参数:
        filename: 输出文件名
        """
        trajectory = self.generate_trajectory()
        
        with open(filename, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        print(f"\n轨迹已保存到: {filename}")
        print(f"轨迹点数: {len(trajectory)}")
    
    def plot_trajectory(self):
        """绘制轨迹曲线"""
        if self.optimized_coeffs is None:
            raise ValueError("请先执行optimize()方法进行优化")
        
        # 生成时间序列
        time_series = np.arange(0, self.T + self.dt, self.dt)
        
        # 计算位置、速度、加速度
        positions = np.zeros((len(time_series), self.n_joints))
        velocities = np.zeros((len(time_series), self.n_joints))
        accelerations = np.zeros((len(time_series), self.n_joints))
        
        for i, t in enumerate(time_series):
            positions[i] = self.compute_position(self.optimized_coeffs, t)
            velocities[i] = self.compute_velocity(self.optimized_coeffs, t)
            accelerations[i] = self.compute_acceleration(self.optimized_coeffs, t)
        
        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 绘制位置
        for j in range(self.n_joints):
            axes[0].plot(time_series, positions[:, j], label=f'joint {j+1}')
        axes[0].set_xlabel('time (s)')
        axes[0].set_ylabel('pos (rad)')
        axes[0].set_title('pos_tra')
        axes[0].grid(True)
        axes[0].legend(loc='upper right', ncol=2)
        
        # 绘制速度
        for j in range(self.n_joints):
            axes[1].plot(time_series, velocities[:, j], label=f'joint {j+1}')
        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('vec (rad/s)')
        axes[1].set_title('vec_tra')
        axes[1].grid(True)
        axes[1].legend(loc='upper right', ncol=2)
        
        # 绘制加速度
        for j in range(self.n_joints):
            axes[2].plot(time_series, accelerations[:, j], label=f'joint {j+1}')
        axes[2].set_xlabel('time (s)')
        axes[2].set_ylabel('acc (rad/s²)')
        axes[2].set_title('acc_tra')
        axes[2].grid(True)
        axes[2].legend(loc='upper right', ncol=2)
        
        plt.tight_layout()
        plt.show()
        
        # 计算末端距离
        distances = []
        for i, t in enumerate(time_series[:len(self.master_positions)]):
            q_slave = self.compute_position(self.optimized_coeffs, t)
            slave_pos = self.forward_kinematics(q_slave,self.Tbase_slave)
            master_pos = self.master_positions[i]
            distances.append(np.linalg.norm(slave_pos - master_pos))
        
        # 绘制末端距离
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_series[:len(distances)], distances, 'b-', linewidth=2, label='distance')
        ax.axhline(y=self.d_min, color='r', linestyle='--', label=f'dia_min ({self.d_min}m)')
        ax.axhline(y=self.d_max, color='g', linestyle='--', label=f'dis_max ({self.d_max}m)')
        ax.fill_between(time_series[:len(distances)], self.d_min, self.d_max, alpha=0.2, color='yellow')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('distance (m)')
        ax.set_title('distance_of_master_slave')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def diagnose_optimization(self):
        """奶奶，用这个函数来诊断问题"""
        
        # 1. 先检查起点距离
        print("🔍 诊断开始...")
        print("="*50)
        
        # 计算起点距离
        q_master_start = np.array(self.master_trajectory[0]['Joint'])
        master_pos_start = self.forward_kinematics(q_master_start, self.Tbase_master)
        slave_pos_start = self.forward_kinematics(self.q_s, self.Tbase_slave)
        start_distance = np.linalg.norm(master_pos_start - slave_pos_start)
        
        print(f"起点距离: {start_distance:.4f} m")
        print(f"距离约束: [{self.d_min}, {self.d_max}]")
        
        if start_distance < self.d_min:
            print("❌ 问题: 起点距离小于最小值!")
        elif start_distance > self.d_max:
            print("❌ 问题: 起点距离大于最大值!")
        else:
            print("✅ 起点距离符合约束")
        
        # 2. 检查终点距离
        q_master_end = np.array(self.master_trajectory[-1]['Joint'])
        master_pos_end = self.forward_kinematics(q_master_end, self.Tbase_master)
        slave_pos_end = self.forward_kinematics(self.q_e, self.Tbase_slave)
        end_distance = np.linalg.norm(master_pos_end - slave_pos_end)
        
        print(f"\n终点距离: {end_distance:.4f} m")
        if end_distance < self.d_min:
            print("❌ 问题: 终点距离小于最小值!")
        elif end_distance > self.d_max:
            print("❌ 问题: 终点距离大于最大值!")
        else:
            print("✅ 终点距离符合约束")
        
        # 3. 检查关节限位
        print("\n关节位置检查:")
        for i in range(6):
            print(f"  关节{i+1}: {self.q_s[i]:.3f} -> {self.q_e[i]:.3f}")
            print(f"    限位: [{self.q_min[i]:.3f}, {self.q_max[i]:.3f}]")
            
            if self.q_s[i] < self.q_min[i] or self.q_s[i] > self.q_max[i]:
                print(f"  ❌ 起点超出限位!")
            if self.q_e[i] < self.q_min[i] or self.q_e[i] > self.q_max[i]:
                print(f"  ❌ 终点超出限位!")
        
        return start_distance, end_distance

def main():
    """主函数示例"""
    print("从臂轨迹规划示例")
    print("=" * 50)
    
    # # 创建示例主臂轨迹文件
    # create_sample_master_trajectory()
    
    # 从臂轨迹参数
    T = 200.0  # 总时间5秒
    dt = 1/125  # 时间步长
    
    # 从臂起始和结束位置（6个关节）
    qs_d=[59.285,17.656,104.646,-3.474,63.511,-17.628]
    qe_d=[31.659,27.489,96.695,-74.691,-65.946,-17.627]
    qs_r=[0.0,0.0,0.0,0.0,0.0,0.0]
    qe_r=[0.0,0.0,0.0,0.0,0.0,0.0]
    #P=Tra.Trajectory(100,[0.0,0.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0,1.0,1.0])
    for i in range(len(qs_d)):
        qs_r[i]=math.radians(qs_d[i])
        qe_r[i]=math.radians(qe_d[i])
    
    # 创建从臂轨迹规划器
    print(f"创建从臂轨迹规划器...")
    print(f"起始位置: {qs_r}")
    print(f"结束位置: {qe_r}")
    
    planner = SlaveTrajectoryOptimizer(
        T=T,
        q_s=qs_r,
        q_e=qe_r,
        Tbase_master=np.array([[1.0,0.0,0.0,0.275],[0.0,1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]),
        Tbase_slave=np.array([[-1.0,0.0,0.0,-0.275],[0.0,-1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]),
        master_trajectory_file='trajectory_data.json',
        d_min=0.1,   # 最小距离20cm
        d_max=0.5,   # 最大距离30cm
        dt=dt
    )
    

    start_dist, end_dist = planner.diagnose_optimization()
    # 执行优化
    success = planner.optimize(
        max_iter=1000,      # 增加迭代次数
        disp=True           # 显示详细信息
    )
    
    if success:
        # 验证约束
        planner.validate_constraints(tolerance=1e-3)
        
        # 保存轨迹
        planner.save_trajectory('slave_trajectory_data.json')
        
        # 绘制轨迹
        try:
            planner.plot_trajectory()
        except:
            print("绘图功能需要matplotlib库，请安装: pip install matplotlib")
    else:
        print("优化失败，无法生成有效轨迹")


if __name__ == "__main__":
    main()